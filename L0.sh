#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 1 --mesh_obj_list 1 --is_pbr true --occlusion true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 5 --mesh_obj_list 5 --is_pbr true --occlusion true -d 0