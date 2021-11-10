#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 8 --mesh_obj_list 8 --is_pbr true --occlusion true -d 0
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 12 --mesh_obj_list 12 --is_pbr true --occlusion true -d 0
