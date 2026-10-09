import math
import numpy as np
from lib.models.DAVLT import build_DAVLT
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.layers.nlp_embedding import nlp_embedding

import torch.nn.functional as F

class DAVLT(BaseTracker):
    def __init__(self, params):
        super(DAVLT, self).__init__(params)
        network = build_DAVLT(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.l_feat = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.nlp_embedding = nlp_embedding()
        self.mem_feat=[]
        self.cur_feat = torch.Tensor().cuda()
        self.q_feat = None
        self.max_score=0

    def initialize(self, image, nlp,  info: dict):
        self.l_feat=self.nlp_embedding([nlp])
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

        self.query=None
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:

            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


    def track(self, image, nlp,  info: dict = None):

        H, W, _ = image.shape

        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # --------- select memory frames ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = [self.memory_frames[0],self.memory_frames[0],self.memory_frames[0]]
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat((self.memory_masks[0],self.memory_masks[0],self.memory_masks[0]), dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=[search.tensors],
                                            ce_template_mask=box_mask_z,  nlp=nlp, l_feat=self.l_feat,q_feat=self.q_feat)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        query_feat=out_dict['query_feature']
        search_feat=out_dict['backbone_feat']
        global_feat = out_dict['global_feat']

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)

        search_feat=self.crop_tokens_from_bbox(search_feat,  pred_boxes[0])
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]


        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)


        # --------- save memory frames and masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                  output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()

        score=out_dict['score']
        if score>self.max_score:
            self.max_score=score
            self.cur_feat=query_feat

        if self.frame_id%600==0:
            self.updata_feature()

        sim =torch.matmul(
            self.l_feat.mean(dim=1).squeeze(0) ,
            global_feat.mean(dim=1).squeeze(0).T ,

        ).squeeze(-1).cpu().item()

        '''sim = F.cosine_similarity(
            query_feat.mean(dim=1),
            search_feat.mean(dim=1),
            dim=-1
        )'''

        if  score> 0.4 and sim < 0.75 and sim>0.5:
            self.q_feat=self.select_by_global_similarity(global_feat)

        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():  # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save memory frames and masks ---------

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:

            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,"sim":sim,"score":out_dict['score']}

    def select_by_global_similarity(self, global_feat, threshold=0.85):
        """
        从特征列表中选择与 global_feat 最相似的特征
        """
        assert isinstance(self.mem_feat, list)
        assert all(isinstance(f, torch.Tensor) for f in self.mem_feat)
        assert isinstance(global_feat, torch.Tensor)

        global_feat = global_feat.reshape(-1)

        max_sim = -1.0
        best_feat = None

        for feat in self.mem_feat:
            feat_flat = feat.reshape(-1)
            sim = F.cosine_similarity(feat_flat.mean(dim=1), global_feat.mean(dim=1), dim=0).item()

            if sim > max_sim:
                max_sim = sim
                best_feat = feat

        if max_sim > threshold:
            return best_feat
        else:
            return None


    def updata_feature(self,min_len=10):

        assert isinstance(self.mem_feat, list)

        tensors = self.mem_feat.copy()

        while len(tensors) > min_len:
            sims = []

            # 计算相邻 tensor 的相似度
            for i in range(len(tensors) - 1):
                t1 = tensors[i].reshape(-1)
                t2 = tensors[i + 1].reshape(-1)
                sim = F.cosine_similarity(t1.mean(dim=1), t2.mean(dim=1), dim=0)
                sims.append(sim.item())

            # 找到相似度最大的相邻对
            max_idx = max(range(len(sims)), key=lambda i: sims[i])

            # 合并这两个 tensor
            merged = tensors[max_idx] + tensors[max_idx + 1]

            # 更新列表
            tensors = (
                    tensors[:max_idx]
                    + [merged]
                    + tensors[max_idx + 2:]
            )
        self.mem_feat.append(self.cur_feat)

        return tensors

    def crop_tokens_from_bbox(self,
            feat,  # (1, 576, 512)
            bbox,  # (cx, cy, w, h), normalized [0,1]
            grid_size=24
    ):
        """
        Returns:
            cropped_feat: (num_tokens, 512)
        """
        assert feat.dim() == 3
        assert feat.shape[0] == 1
        assert feat.shape[1] == grid_size * grid_size

        cx, cy, w, h = bbox

        # 1. reshape to (1, 24, 24, 512)
        feat = feat.view(1, grid_size, grid_size, -1)

        # 2. bbox corners (normalized)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # clamp to [0,1]
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(1., x2), min(1., y2)

        # 3. map to token indices
        ix1 = int(torch.floor(torch.tensor(x1 * grid_size)))
        iy1 = int(torch.floor(torch.tensor(y1 * grid_size)))
        ix2 = int(torch.ceil(torch.tensor(x2 * grid_size)))
        iy2 = int(torch.ceil(torch.tensor(y2 * grid_size)))

        # safety clamp
        ix1 = max(0, min(ix1, grid_size - 1))
        iy1 = max(0, min(iy1, grid_size - 1))
        ix2 = max(ix1 + 1, min(ix2, grid_size))
        iy2 = max(iy1 + 1, min(iy2, grid_size))

        # 4. crop
        cropped = feat[:, ix1:ix2,iy1:iy2,  :]  # (1, h, w, 512)

        # 5. flatten tokens
        cropped = cropped.reshape(-1, feat.shape[-1]).unsqueeze(0)  # (N, 512)

        return cropped

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.array(list(range(num_segments))) * dur + dur // 2
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)[1:]
        indexes = np.insert(indexes,0,0)

        select_frames, select_masks = [], []

        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())

        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return DAVLT

