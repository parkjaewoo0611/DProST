#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 2 --mesh_obj_list 2 --is_pbr true --occlusion true -d 1 -c saved/models/ProjectivePose/1109_172853/config.json -r saved/models/ProjectivePose/1109_172853/checkpoint-epoch1300.pth
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 2 --mesh_obj_list 2 --is_pbr true --occlusion false -d 1
