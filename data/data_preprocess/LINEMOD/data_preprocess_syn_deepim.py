import os
import json
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from PIL import Image

class Preprocessor():
    def __init__(self, mode, name):
        if name == 'lm':
            self.DATA_PATH = 'data/LINEMOD/syn'
            self.obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
            self.N = 1000
        else:
            self.DATA_PATH = 'data/OCCLUSION/syn'
            self.obj_list = [1, 5, 6, 8, 9, 10, 11, 12]
            self.N = 10000
    def build(self):
        POSE_LIST = []
        OBJ_PATH = glob.glob(f'{self.DATA_PATH}/**')
        OBJ_PATH = [obj_path for obj_path in OBJ_PATH if LM_class2idx[obj_path.split('/')[-1]] in self.obj_list]
        for obj_path in OBJ_PATH:
            POSE_LIST.extend(glob.glob(f'{obj_path}/*.txt')[:self.N])
        self.FILE_LIST = [pose_path.split('-')[0] for pose_path in POSE_LIST]
        return self

    def mask_to_bbox(self, mask):
        coordinate = np.where(mask)
        x_min = coordinate[1].min()
        x_max = coordinate[1].max()
        y_min = coordinate[0].min()
        y_max = coordinate[0].max()
        return [x_min, y_min, x_max, y_max]

    def load(self, fname):
        img_name = f'{fname}-color.png'
        depth_name = f'{fname}-depth.png'
        label_name = f'{fname}-pose.txt'

        with open(label_name, 'r') as f:
            label = f.read().splitlines()
        
        obj_id = LM_class2idx[fname.split('/')[-2]]
        RT = label_to_RT(label[1:])

        K = K_LM.copy()

        depth = np.array(Image.open(depth_name))
        mask = depth.astype(bool)
        bbox_obj = self.mask_to_bbox(mask)
        print(np.linalg.norm(RT[:3, 3]))
        if np.linalg.norm(RT[:3, 3]) > 10:
            print('issue!')
        obj = {
                'RT' : RT,
                'K' : K,
                'bbox_obj' : bbox_obj,
                'image' : img_name,
                'depth' : depth_name,
                'depth_scale': 1.0,
                'mask' : None,
                'visib_fract' : 1.0,
                'obj_id' : obj_id
                }
        return obj

    def __getitem__(self, index):
        obj = self.load(self.FILE_LIST[index])
        return obj

    def __len__(self):
        return len(self.FILE_LIST)


def label_to_RT(text):
    RT = []
    for i, line in enumerate(text):
        RT.append([float(comp) for comp in line.split(' ')])
    const = np.array([0, 0, 0, 1], np.float32)[np.newaxis, ...]
    RT = np.concatenate((np.array(RT), const), axis=-2)
    return RT


FX = 572.4114
FY = 573.57043
PX = 325.2611
PY = 242.04899

K_LM = np.array([[FX,  0, PX],
                 [ 0, FY, PY],
                 [ 0,  0,  1]])

LM_class2idx = {
    "ape" : 1,
    "benchvise" : 2,
    #'bowl' : 3,
    "camera" : 4,
    "can" : 5,
    "cat": 6,
    #"cup" : 7,
    "driller" : 8,
    "duck" : 9,
    "eggbox" : 10,
    "glue" : 11,
    "holepuncher" : 12,
    "iron" : 13,
    "lamp" : 14,
    "phone" : 15,
}