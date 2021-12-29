import os
import json
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from PIL import Image

LM_idx2class = {
    1: "ape",
    2: "benchvise",
    #3: 'bowl',
    4: "camera",
    5: "can",
    6: "cat",
    #7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

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

def RT_to_TCO(R, T):
    TCO = np.concatenate((R, T), axis=-1)
    const = np.zeros_like(TCO)[..., [0], :]
    const[..., 0, -1] = 1
    TCO = np.concatenate((TCO, const), axis=-2)
    return TCO

class Preprocessor():
    def __init__(self, mode):
        self.obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.DATA_PATH = '../Dataset/synthetic/LINEMOD/renders'      #TODO: move in Dataset/LINEMOD

    def build(self):
        LABEL_LIST = glob.iglob(f'{self.DATA_PATH}/**/*.pkl', recursive=True)
        self.FILE_LIST = [label.split('_')[0] for label in LABEL_LIST]
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
        obj={}
        ratio = 1
        K = label['K']
        K = K * ratio
        K[-1, -1] = 1

        TCO = label['RT']
        TCO = RT_to_TCO(TCO[:3, :3], TCO[:3, [3]])
        mask = np.array(Image.open(mask_name))
        mask = (mask / mask.max() * 255).astype(np.uint8)
        bbox_obj = self.mask_to_bbox(mask)
        bbox_obj = [loc * ratio for loc in bbox_obj]
        visib_fract = 1.0
        obj_id = LM_class2idx[file.split('/')[-2]]
        obj = {
                'TCO' : TCO,
                'K' : K,
                'bbox_obj' : bbox_obj,
                'image' : img_name,
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

