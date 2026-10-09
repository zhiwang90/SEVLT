import glob
import json
import numpy as np
import os
import torch
import glob
import pandas
import six
from .base_video_dataset import BaseVideoDataset
import numpy as np
from collections import OrderedDict
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings



class MGIT(BaseVideoDataset):
    r"""`MGIT <http://videocube.aitestunion.com>`_ Dataset.

    Publication:
        ``A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship``, S. Hu, D. Zhang, M. Wu, X. Feng, X. Li, X. Zhao, K. Huang
        Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2023

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        split (string, optional): Specify ``train``, ``val`` or ``test``
            subset of MGIT.
    """

    def __init__(self,root=None,  version='tiny',split=None, image_loader=jpeg4py_loader,):
        root = env_settings().mgit_dir if root is None else root
        super(MGIT, self).__init__('mgit', root, image_loader)
        assert split in ['train', 'val', 'test'], 'Unknown subset.'

        self.base_path = "/home/zyh/datasets/MGIT/MGIT-Test"
        self.split = split
        self.version = version  # temporarily, the toolkit only support tiny version of MGIT

        f = open(os.path.join('/home/zyh/DAVLT/lib/train/data_specs', 'mgit.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        self.sequence_list = self.infos[self.split]
        detailed_anno_path = env_settings().lang_anno_dir
        if split in ['train', 'val', 'test']:
            self.seq_dirs = [os.path.join(root, 'MGIT-Train', s, 'frame_{}'.format(s)) for s in self.sequence_list]
            self.anno_files = [os.path.join(root, 'attribute', 'groundtruth', '{}.txt'.format(s)) for s in
                               self.sequence_list]
            self.detailed_anno_path = [os.path.join(detailed_anno_path, 'train', 'mgit_train_concise', '{}.txt'.format(s)) for s in
                                  self.sequence_list]

    def get_name(self):
        return 'MGIT'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info


    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.read_attr(os.path.join(self.root, s))['class']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]



    def read_nlp(self, seq_id):

        nlp_file = self.detailed_anno_path[seq_id]
        with open(nlp_file) as f:
            nlp = [line.strip().split(maxsplit=1)[1] for line in f if line.strip()]
        return nlp

    def _read_bb_anno(self, seq_id):

        bb_anno_file = self.anno_files[seq_id]
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)


    def get_sequence_info(self, seq_id):
        seq_path = self.seq_dirs[seq_id]
        seq_name = seq_path.split('/')[-2]
        bbox = self._read_bb_anno(seq_id)
        nlp = self.read_nlp(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'nlp': nlp, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        images = sorted(glob.glob(os.path.join(seq_path,  '*')))

        return self.image_loader(images[frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None, get_his=0):
        seq_path = self.seq_dirs[seq_id]
        object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, object_meta