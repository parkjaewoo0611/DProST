import os
import json
import numpy as np

class Preprocessor():
    def __init__(self, mode):
        self.obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.DATA_PATH = '../Dataset/LINEMOD/data_pbr'
        self.INDEX_PATH = '../Dataset/LINEMOD/index'

    def build(self):
        self.camera = self.load_camera_files(self.DATA_PATH)
        self.gt_info = self.load_gt_info_files(self.DATA_PATH)
        self.visible_fract_thr = 0.0
        self.LABEL_LIST = self.scene_img_obj_list(self.gt_info, self.obj_list)
        return self


    def load_camera_files(self, data_path):
        scene_list = os.listdir(data_path)
        scene_camera_dict = {}
        for scene_id in scene_list:
            with open(os.path.join(data_path, scene_id, 'scene_camera.json')) as f:
                scene_camera = json.load(f)
            im_camera_dict = {}
            for im_id in scene_camera.keys():
                K = np.array(scene_camera[im_id]['cam_K']).reshape(3, 3)
                depth_scale= np.array(scene_camera[im_id]['depth_scale'])
                im_camera_dict[int(im_id)] = {'K' : K, 
                                              'depth_scale' : depth_scale}
            scene_camera_dict[int(scene_id)] = im_camera_dict
        return scene_camera_dict

    def load_gt_info_files(self, data_path):
        scene_list = os.listdir(data_path)
        scene_dict = {}
        for scene_id in scene_list:
            with open(os.path.join(data_path, scene_id, 'scene_gt.json')) as f:
                scene_gt = json.load(f)
            with open(os.path.join(data_path, scene_id, 'scene_gt_info.json')) as f:
                scene_in = json.load(f)
            im_dict = {}
            for im_id in scene_gt.keys():
                inst_dict = {}
                for i, (inst_gt, inst_in) in enumerate(zip(scene_gt[im_id], scene_in[im_id])):
                    TCO = RT_to_TCO(np.array(inst_gt['cam_R_m2c']).reshape(3, 3),
                                    np.array(inst_gt['cam_t_m2c']).reshape(3, 1))
                    obj_id = inst_gt['obj_id']
                    inst_in['bbox_obj'][2] += inst_in['bbox_obj'][0]
                    inst_in['bbox_obj'][3] += inst_in['bbox_obj'][1]
                    inst_dict[i] = {'TCO' : TCO,
                                    'obj_id' : obj_id,
                                    'bbox_obj' : inst_in['bbox_obj'],
                                    'visib_fract' : inst_in['visib_fract']}
                im_dict[int(im_id)] = inst_dict
            scene_dict[int(scene_id)] = im_dict
        return scene_dict        

    def scene_img_obj_list(self, gt_info, obj_list):
        LABEL_LIST = []
        for scene_id in gt_info.keys():
            for im_id in gt_info[scene_id].keys():
                for inst_id in gt_info[scene_id][im_id].keys():
                    obj_id = gt_info[scene_id][im_id][inst_id]['obj_id']
                    if obj_id in obj_list:
                        if gt_info[scene_id][im_id][inst_id]['visib_fract'] > self.visible_fract_thr:
                            LABEL_LIST.append({'scene_id' : int(scene_id),
                                               'im_id' : int(im_id),
                                               'inst_id' : int(inst_id)})
        return LABEL_LIST


    def load(self, scene_id, im_id, inst_id):
        img_name = self.load_image(scene_id, im_id)
        mask_name = self.load_mask(scene_id, im_id, inst_id)
        ratio = 1
        K = self.camera[scene_id][im_id]['K']
        K = K * ratio
        K[-1, -1] = 1
        depth_scale = self.camera[scene_id][im_id]['depth_scale']
        TCO = self.gt_info[scene_id][im_id][inst_id]['TCO']
        bbox_obj = self.gt_info[scene_id][im_id][inst_id]['bbox_obj']
        bbox_obj = [loc * ratio for loc in bbox_obj]
        visib_fract = self.gt_info[scene_id][im_id][inst_id]['visib_fract']
        obj_id = self.gt_info[scene_id][im_id][inst_id]['obj_id']
        obj = {
                'TCO' : TCO,
                'K' : K,
                'bbox_obj' : bbox_obj,
                'image' : img_name,
                'mask' : mask_name,
                'depth_scale' : depth_scale,
                'visib_fract' : visib_fract,
                'scene_id' : scene_id,
                'im_id' : im_id,
                'obj_id' : obj_id
                }
        return obj

    def load_image(self, scene_id, im_id):
        scene_id = "{0:0=6d}".format(scene_id)
        im_id = "{0:0=6d}".format(im_id)
        name = os.path.join(self.DATA_PATH, scene_id, 'rgb', im_id)+'.jpg'
        return name

    def load_mask(self, scene_id, im_id, inst_id):
        scene_id = "{0:0=6d}".format(scene_id)
        im_id = "{0:0=6d}".format(im_id)
        inst_id = "{0:0=6d}".format(inst_id)
        name = os.path.join(self.DATA_PATH, scene_id, 'mask', im_id + '_' + inst_id)+'.png'
        return name

    def __getitem__(self, index):
        obj = self.load(self.LABEL_LIST[index]['scene_id'], 
                        self.LABEL_LIST[index]['im_id'], 
                        self.LABEL_LIST[index]['inst_id'])
        return obj

    def __len__(self):
        return len(self.LABEL_LIST)


def RT_to_TCO(R, T):
    TCO = np.concatenate((R, T), axis=-1)
    const = np.zeros_like(TCO)[..., [0], :]
    const[..., 0, -1] = 1
    TCO = np.concatenate((TCO, const), axis=-2)
    return TCO


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