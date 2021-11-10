#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 9 --mesh_obj_list 9 --is_pbr true --occlusion true -d 1
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 10 --mesh_obj_list 10 --is_pbr true --occlusion true -d 1
