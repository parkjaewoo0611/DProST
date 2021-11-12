#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 4 --mesh_obj_list 4 --is_pbr true --occlusion true -d 2 -c saved/models/ProjectivePose/1109_212421/config.json -r saved/models/ProjectivePose/1109_212421/checkpoint-epoch1500.pth
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 4 --mesh_obj_list 4 --is_pbr true --occlusion false -d 2
