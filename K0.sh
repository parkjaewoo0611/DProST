#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr true --occlusion true -d 0 -c saved/models/ProjectivePose/1109_235238/config.json -r saved/models/ProjectivePose/1109_235238/checkpoint-epoch900.pth
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr true --occlusion false -d 0
