#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 1 --mesh_obj_list 1 --is_pbr true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 6 --mesh_obj_list 6 --is_pbr true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 8 --mesh_obj_list 8 --is_pbr true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 1 --mesh_obj_list 1 --is_pbr false -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr false -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 6 --mesh_obj_list 6 --is_pbr false -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 8 --mesh_obj_list 8 --is_pbr false -d 0