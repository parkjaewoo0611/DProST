#!/bin/bash
# LINEMOD linking
mkdir -p data/LINEMOD
ln -sf /home/jw/D/Pose/Dataset/VOCdevkit/VOC2012/JPEGImages ./data/LINEMOD/background         # background image folder  
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/train data/LINEMOD/train                                # train dataset folder
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/train_pbr data/LINEMOD/pbr                              # BOP pbr dataset folder
ln -sf /home/jw/D/Pose/Dataset/LINEMOD/DeepIM_syn data/LINEMOD/syn                            # synthetic image folder from DeepIM
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/test data/LINEMOD/test                                  # test dataset folder
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/models data/LINEMOD/models                              # object ply folder
ln -sf /home/jw/D/Pose/Dataset/LINEMOD/index data/LINEMOD/index                               # index folder of real dataset train/test
ln -sf /home/jw/D/Pose/Dataset/LINEMOD/test_bboxes data/LINEMOD/test_bboxes                   # tesbboxes from CDPN

# LINEMOD-OCCLUSION linking
mkdir -p data/OCCLUSION
ln -sf /home/jw/D/Pose/Dataset/VOCdevkit/VOC2012/JPEGImages ./data/OCCLUSION/background         # background image folder  
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/test data/OCCLUSION/train                                 # train dataset folder
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/train_pbr data/OCCLUSION/pbr                              # BOP pbr dataset folder
ln -sf /home/jw/D/Pose/Dataset/LINEMOD/DeepIM_syn data/OCCLUSION/syn                            # synthetic image folder from DeepIM
ln -sf /home/jw/D/Pose/Dataset/BOP/lmo/test data/OCCLUSION/test                                 # test dataset folder
ln -sf /home/jw/D/Pose/Dataset/BOP/lm/models data/OCCLUSION/models                              # object ply folder
ln -sf /home/jw/D/Pose/Dataset/OCCLUSION/index data/OCCLUSION/index                             # index folder of real dataset train/test
ln -sf /home/jw/D/Pose/Dataset/OCCLUSION/test_bboxes data/OCCLUSION/test_bboxes                 # tesbboxes from CDPN
