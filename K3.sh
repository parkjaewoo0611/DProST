#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr true --occlusion true -d 3 -c saved/models/ProjectivePose/1110_123406/config.json -r saved/models/ProjectivePose/1110_123406/checkpoint-epoch300.pth
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr true --occlusion false -d 3
