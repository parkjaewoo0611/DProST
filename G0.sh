#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 6 --mesh_obj_list 6 --is_pbr false -d 0
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 8 --mesh_obj_list 8 --is_pbr false -d 0
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr false -d 0
