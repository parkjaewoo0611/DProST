import os
import json
import numpy as np
from collections import defaultdict

class Preprocessor():
    def __init__(self, mode):
        self.obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode
        self.DATA_PATH = '../Dataset/LINEMOD/data'
        self.INDEX_PATH = '../Dataset/LINEMOD/index'
        self.BBOX_PATH = '../Dataset/LINEMOD/test_bboxes'

    def build(self):
        self.camera = self.load_camera_files(self.DATA_PATH)
        self.gt_info = self.load_gt_info_files(self.DATA_PATH)
        self.visible_fract_thr = 0.0
        self.LABEL_LIST = self.scene_img_obj_list(self.gt_info, self.obj_list)
        if self.mode == 'test':
            self.bbox_yolo_faster = self.load_test_bbox()
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
                obj_dict = {}
                for inst_gt, inst_in in zip(scene_gt[im_id], scene_in[im_id]):
                    TCO = RT_to_TCO(np.array(inst_gt['cam_R_m2c']).reshape(3, 3),
                                    np.array(inst_gt['cam_t_m2c']).reshape(3, 1))
                    obj_id = inst_gt['obj_id']
                    inst_in['bbox_obj'][2] += inst_in['bbox_obj'][0]
                    inst_in['bbox_obj'][3] += inst_in['bbox_obj'][1]
                    obj_dict[int(obj_id)] = {'TCO' : TCO,
                                             'obj_id' : obj_id,
                                             'bbox_obj' : inst_in['bbox_obj'],
                                             'visib_fract' : inst_in['visib_fract']}
                im_dict[int(im_id)] = obj_dict
            scene_dict[int(scene_id)] = im_dict
        return scene_dict        

    def load_test_bbox(self):
        with open(os.path.join(self.BBOX_PATH, 'bbox_faster_all.json')) as f:
            faster = json.load(f)
        with open(os.path.join(self.BBOX_PATH, 'bbox_yolov3_all.json')) as f:
            yolo = json.load(f)
        scene_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        keys = list(faster.keys())
        for key in keys:
            scene_id = int(key.split('/')[0])
            im_id = int(key.split('/')[1])
            obj_id = faster[key][0]['obj_id']
            f_bbox = faster[key][0]['bbox_est']
            y_bbox = yolo[key][0]['bbox_est']
            f_bbox[2] += f_bbox[0]
            f_bbox[3] += f_bbox[1]
            y_bbox[2] += y_bbox[0]
            y_bbox[3] += y_bbox[1]
            scene_dict[scene_id][im_id][obj_id] = {
                'bbox_faster': f_bbox,
                'bbox_yolo': y_bbox
            }
        return scene_dict

    def scene_img_obj_list(self, gt_info, obj_list):
        CHECK_LIST = []
        index_file_list = os.listdir(self.INDEX_PATH)
        for index_file in index_file_list:
            obj_id = LM_class2idx[index_file.split('_')[0]]
            if index_file.split('_')[-1].split('.')[0] == self.mode:
                with open(os.path.join(self.INDEX_PATH, index_file)) as f:
                    indexes = f.read().splitlines()
                for index in indexes:
                    scene_id, im_id = index.split('/')
                    CHECK_LIST.append(int(scene_id)*1e10 + (int(im_id)-1)*1e5 + int(obj_id))
        LABEL_LIST = []
        for scene_id in gt_info.keys():
            for im_id in gt_info[scene_id].keys():
                for obj_id in gt_info[scene_id][im_id].keys():
                    if obj_id in obj_list:
                        if gt_info[scene_id][im_id][obj_id]['visib_fract'] > self.visible_fract_thr:
                            if int(scene_id)*1e10 + int(im_id)*1e5 + int(obj_id) in CHECK_LIST:
                                LABEL_LIST.append({'scene_id' : int(scene_id),
                                                'im_id' : int(im_id),
                                                'obj_id' : int(obj_id)})
        return LABEL_LIST

    def load(self, scene_id, im_id, obj_id):
        img_name = self.load_image(scene_id, im_id)
        depth_name = self.load_depth(scene_id, im_id)
        mask_name = self.load_mask(scene_id, im_id, obj_id)
        ratio = 1
        K = self.camera[scene_id][im_id]['K']
        K = K * ratio
        K[-1, -1] = 1
        depth_scale = self.camera[scene_id][im_id]['depth_scale']
        TCO = self.gt_info[scene_id][im_id][obj_id]['TCO']
        bbox_obj = self.gt_info[scene_id][im_id][obj_id]['bbox_obj']
        bbox_obj = [loc * ratio for loc in bbox_obj]
        visib_fract = self.gt_info[scene_id][im_id][obj_id]['visib_fract']
        obj = {
                'TCO' : TCO,
                'K' : K,
                'bbox_obj' : bbox_obj,
                'image' : img_name,
                'depth': depth_name,
                'mask' : mask_name,
                'depth_scale' : depth_scale,
                'visib_fract' : visib_fract,
                'scene_id' : scene_id,
                'im_id' : im_id,
                'obj_id' : obj_id
                }
        if self.mode == 'test':
            bbox_faster = self.bbox_yolo_faster[scene_id][im_id][obj_id]['bbox_faster']
            bbox_faster = [loc * ratio for loc in bbox_faster]
            bbox_yolo = self.bbox_yolo_faster[scene_id][im_id][obj_id]['bbox_yolo']
            bbox_yolo = [loc * ratio for loc in bbox_yolo]
            obj['bbox_faster'] = bbox_faster
            obj['bbox_yolo'] = bbox_yolo
        return obj

    def load_image(self, scene_id, im_id):
        scene_id = "{0:0=6d}".format(scene_id)
        im_id = "{0:0=6d}".format(im_id)
        name = os.path.join(self.DATA_PATH, scene_id, 'rgb', im_id)+'.png'
        return name

    def load_depth(self, scene_id, im_id):
        scene_id = "{0:0=6d}".format(scene_id)
        im_id = "{0:0=6d}".format(im_id)
        name = os.path.join(self.DATA_PATH, scene_id, 'depth', im_id)+'.png'
        return name
    
    def load_mask(self, scene_id, im_id, obj_id):
        scene_id = "{0:0=6d}".format(scene_id)
        im_id = "{0:0=6d}".format(im_id)
        name = os.path.join(self.DATA_PATH, scene_id, 'mask', im_id + '_' + '000000')+'.png'
        return name

    def __getitem__(self, index):
        obj = self.load(self.LABEL_LIST[index]['scene_id'], 
                        self.LABEL_LIST[index]['im_id'], 
                        self.LABEL_LIST[index]['obj_id'])
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