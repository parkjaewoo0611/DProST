#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr true --occlusion true -d 1 -c saved/models/ProjectivePose/1106_143925/config.json -r saved/models/ProjectivePose/1106_143925/checkpoint-epoch2900.pth
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr true --occlusion false -d 1
