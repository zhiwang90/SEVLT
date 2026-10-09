import os
import os.path
import numpy as np
import torch
import glob
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class TNL2k(BaseVideoDataset):


    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().tnl2k_dir if root is None else root
        super().__init__('tnl2k', root, image_loader)

        # all folders inside the root
        self.detailed_anno_path = os.path.join(env_settings().lang_anno_dir, "train/tnl2k_train_concise")
        self.sequence_list = self._get_sequence_list()

        self.sequence_meta_info = self._load_meta_info()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_seq_per_class()


    def get_name(self):
        return 'tnl2k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

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

    def _get_sequence_list(self):
        sequence_list = []

        for seq in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, seq)):
                sequence_list.append(seq)

        return sequence_list


    def read_attr(self, seq_path):

        attr_file = os.path.join(seq_path, "shuxing.txt")
        if not os.path.exists(attr_file):
            attr = {'class':'None','color':'None','action':'None','location':'None'}
            #print(nlp_file)
            return attr

        with open(attr_file, 'r') as f:
            shuxing = f.readlines()
            #key, value = shuxing.split(':')
            for i in shuxing:
                if i =='\n':
                    shuxing.remove(i)
            attr=dict(item.strip().split(":") for item  in shuxing)
            return attr

    def read_nlp(self, seq_name):

        nlp_file = os.path.join(self.detailed_anno_path, seq_name + ".txt")
        with open(nlp_file) as f:
            nlp = [line.strip().split(maxsplit=1)[1] for line in f if line.strip()]
        return nlp

    def _read_bb_anno(self, seq_path):

        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)


    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = seq_path.split('/')[-1]
        bbox = self._read_bb_anno(seq_path)
        nlp = self.read_nlp(seq_name)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'nlp':nlp , 'visible': visible}


    def _get_frame(self, seq_path, frame_id):
        images = sorted(glob.glob(os.path.join(seq_path, 'imgs', '*')))
        return self.image_loader(images[frame_id])



    def get_frames(self, seq_id, frame_ids, anno=None,get_his=0):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta

