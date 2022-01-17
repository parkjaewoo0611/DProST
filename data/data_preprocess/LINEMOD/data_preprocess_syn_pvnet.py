#### USE DEEPIM syn instead ####

import os
import json
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from PIL import Image

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
FX = 572.4114
FY = 573.57043
PX = 325.2611
PY = 242.04899

K_LM = np.array([[FX,  0, PX],
                 [ 0, FY, PY],
                 [ 0,  0,  1]])

def R_T_to_RT(R, T):
    RT = np.concatenate((R, T), axis=-1)
    const = np.zeros_like(RT)[..., [0], :]
    const[..., 0, -1] = 1
    RT = np.concatenate((RT, const), axis=-2)
    return RT

class Preprocessor():
    def __init__(self, mode, name):
        self.DATA_PATH = 'data/LINEMOD/syn_pvnet_lmo'      
        self.obj_list = [1, 5, 6, 8, 9, 10, 11, 12]

    def build(self):
        self.FILE_LIST = []
        for folder in list(glob.glob(f'{self.DATA_PATH}/*')):
            obj_id = LM_class2idx[folder.split('/')[-1]]
            if obj_id in self.obj_list:
                LABEL_LIST = glob.glob(f'{folder}/*.jpg')
                self.FILE_LIST.extend([label[:-4] for label in LABEL_LIST])
        return self

    def mask_to_bbox(self, mask):
        coordinate = np.where(mask)
        x_min = coordinate[1].min()
        x_max = coordinate[1].max()
        y_min = coordinate[0].min()
        y_max = coordinate[0].max()
        return [x_min, y_min, x_max, y_max]

    def load(self, file):
        img_name = f'{file}.jpg'
        mask_name = f'{file}_depth.png'
        label_name = f'{file}_RT.pkl'
        with open(label_name, 'rb') as f:
            label = pickle.load(f)

        RT = label['RT']
        RT = R_T_to_RT(RT[:3, :3], RT[:3, [3]])

        K = K_LM.copy()

        mask = np.array(Image.open(mask_name))
        mask = (mask / mask.max() * 255).astype(np.uint8)
        bbox_obj = self.mask_to_bbox(mask)

        visib_fract = 1.0
        obj_id = LM_class2idx[file.split('/')[-2]]
        obj = {
                'RT' : RT,
                'K' : K,
                'bbox_obj' : bbox_obj,
                'image' : img_name,
                'depth' : None,
                'depth_scale': None,
                'mask' : mask_name,
                'visib_fract' : visib_fract,
                'obj_id' : obj_id
                }
        return obj

    def __getitem__(self, index):
        obj = self.load(self.FILE_LIST[index])
        return obj

    def __len__(self):
        return len(self.FILE_LIST)

