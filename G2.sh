#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 14 --mesh_obj_list 14 --is_pbr true --occlusion true -d 2
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 14 --mesh_obj_list 14 --is_pbr true --occlusion false -d 2