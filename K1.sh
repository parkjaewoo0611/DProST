#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 1 --mesh_obj_list 1 --is_pbr false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 2 --mesh_obj_list 2 --is_pbr false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 4 --mesh_obj_list 4 --is_pbr false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr false -d 1