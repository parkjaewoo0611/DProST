#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr true -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr true -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 11 --mesh_obj_list 11 --is_pbr true -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 12 --mesh_obj_list 12 --is_pbr true -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr false -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr false -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 11 --mesh_obj_list 11 --is_pbr false -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 12 --mesh_obj_list 12 --is_pbr false -d 1