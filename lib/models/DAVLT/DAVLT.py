import math
import os
from typing import List
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
import numpy as np
from einops import rearrange
from lib.models.layers.nlp_embedding import nlp_embedding

from lib.models.layers.head import build_box_head
from lib.models.DAVLT.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.DAVLT.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.DAVLT.itpn import fast_itpn_base_3324_patch16_224
from lib.models.DAVLT.HiViT import hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh


class DAVLT(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=2,  pretrain = None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.nlp_embedding = nlp_embedding()

        self.max_size = 10
        self.memory = []

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)
        self.layer_norm3 = nn.LayerNorm(512)


        self.init_querys = nn.Parameter(torch.zeros(1, 24, 512))
        self.fusion_fc1 = nn.Linear(512, 512)


        self.fusion_visual_language = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0,
            batch_first=True)

        self.fusion_querys = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0,
            batch_first=True)

        self.fusion_visual_querys = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0,
            batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512)
        )
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                nlp=None,
                l_feat=None,
                return_last_attn=False,
                flag=True,
                q_feat=None,
                ):
        assert isinstance(search, list), "The type of search is not List"

        if isinstance(nlp, list):
            nlp = [list(group) for group in zip(*nlp)]
            bs = len(nlp)
            nlp = sum(nlp, [])
        else:
            bs=1
        out_dict = []
        if self.training is True:
            flag=np.random.choice([True,False])
            self.track_query = None
        #flag = True

        if l_feat is None:
            lang_features = self.nlp_embedding(nlp).view(bs, -1, 24, 512)
        else:
            lang_features = l_feat.view(bs, -1, 24, 512)

        for i in range(len(search)):
            lang_feature = lang_features[:, i, :, :]
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        temporal_query=self.track_query,
                                        token_len=self.token_len)
            feat_last = x

            if isinstance(x, list):
                feat_last = x[-1]

            temp_feat = feat_last[:, self.token_len:-self.feat_len_s]
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2)).contiguous()  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)

            # Forward head
            out = self.forward_head(opt, lang_feature, flag,q_feat)

            if self.backbone.add_cls_token:
                self.track_query = (x[:, :1].clone()).detach()  # stop grad  (B, N, C)


            out.update(aux_dict)
            out['backbone_feat'] = x[:, -self.feat_len_s:]
            out['global_feat'] = x[:, 0].unsqueeze(1)
            out['lang_feat'] = lang_features[:,i,:,:]


            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, lang_feature, flag,q_feat, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        final_query = None
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)


        sem_querys = self.init_querys.expand(bs, -1, -1).contiguous()
        fused_querys = self.fusion_querys(
                    query=sem_querys,
                    key=lang_feature,
                    value=lang_feature,
                )[0]
        fused_querys =self.layer_norm1(fused_querys)
        att_feat = torch.Tensor().cuda()

        weight=self.fusion_fc1(lang_feature)

        for i in range(bs):
            single_opt = opt_feat[i].unsqueeze(0)
            att_res = con_att(weight[i], single_opt)
            att_feat = torch.cat([att_feat, att_res], 0)
        if q_feat is None:
            att_feat = att_feat + opt_feat
            att_feat = att_feat.view(bs, C, -1).transpose(-1, -2).contiguous()

            final_query = self.fusion_visual_querys(
                        query=fused_querys,
                        key=att_feat,
                        value=att_feat
                    )[0]

            final_query=self.ffn(final_query)+fused_querys

            final_query = self.layer_norm2(final_query)
        else:
            final_query=q_feat
        opt_feat = opt_feat.view(bs, C, -1).transpose(-1, -2).contiguous()

        if flag is True:
            fused_feat = self.fusion_visual_language(
                query=opt_feat,
                key=final_query,
                value=final_query,
            )[0]
            fused_feat = self.layer_norm3(fused_feat)
            opt_feat = opt_feat + fused_feat
            opt_feat = opt_feat.transpose(-1, -2).contiguous()

        else:

            fused_feat = self.fusion_visual_language(
                query=opt_feat,
                key=lang_feature,
                value=lang_feature,
            )[0]
            fused_feat = self.layer_norm3(fused_feat)
            opt_feat = opt_feat + fused_feat
            opt_feat = opt_feat.transpose(-1, -2).contiguous()

        opt_feat = opt_feat.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'score': score_map.max(),
                   'query_feature': final_query
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)

            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)

            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'score': score_map_ctr.max(),
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'query_feature': final_query}

            return out
        else:
            raise NotImplementedError


def con_att(weight, input):
    weight = weight.mean(dim=0).view(1, 512, 1, 1)
    out = F.conv2d(input, weight, stride=1, padding=0)
    return out.repeat(1, 512, 1, 1)


def build_DAVLT(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.BACKBONE_FILE and ('OSTrack' not in cfg.MODEL.BACKBONE_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.BACKBONE_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,
                                         )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        # backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)


    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)


    model = DAVLT(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )
    if cfg.MODEL.PRETRAIN_FILE:
        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)

        model_pretrained = torch.load(pretrained_file, map_location="cpu")

        state_dict = model_pretrained['net']
        model.load_state_dict(state_dict, strict=False)
    return model
