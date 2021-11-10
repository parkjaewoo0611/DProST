#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 15 --mesh_obj_list 15 --is_pbr true --occlusion true -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 15 --mesh_obj_list 15 --is_pbr true --occlusion false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr true --occlusion false -d 1