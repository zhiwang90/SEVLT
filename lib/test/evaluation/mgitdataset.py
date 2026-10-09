import glob
import json
import os
from lib.train.admin import env_settings
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import pandas as pd

class MGITDataset(BaseDataset):

    def __init__(self,version='tiny'):
        super().__init__()

        self.base_path = self.env_settings.mgit_path
        self.version = version
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.sequence_list


    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute','groundtruth', sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/{}/{}/frame_{}'.format(self.base_path, 'MGIT-Test',sequence_name ,sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        nlp_file = '{}/{}/{}/{}.json'.format(self.base_path, 'attribute', 'description', sequence_name)

        with open(nlp_file, "r") as f:
            data = f.read().replace("NaN", "null")
        cfg = json.loads(data)
        nlp = cfg["action"]["action_1"]["description"]


        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'mgit', ground_truth_rect.reshape(-1, 4),nlp=nlp)


    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        f = open('/data3/hly/DAVLT/lib/train/data_specs/mgit.json', 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()
        sequence_list = self.infos['test']
        return sequence_list
