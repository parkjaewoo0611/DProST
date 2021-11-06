#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 13 --mesh_obj_list 13 --is_pbr false -d 2
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 14 --mesh_obj_list 14 --is_pbr false -d 2
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 15 --mesh_obj_list 15 --is_pbr false -d 2
