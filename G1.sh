#!/bin/bash
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 11 --mesh_obj_list 11 --is_pbr false -d 1
python train.py --data_dir ../Dataset/LINEMOD --mesh_dir ../Dataset/LINEMOD/models --data_obj_list 12 --mesh_obj_list 12 --is_pbr false -d 1
