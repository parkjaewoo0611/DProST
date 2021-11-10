#!/bin/bash
python train.py --data_dir ../Dataset/OCCLUSION --mesh_dir ../Dataset/OCCLUSION/models --data_obj_list 11 --mesh_obj_list 11 --is_pbr true --occlusion true -d 0
