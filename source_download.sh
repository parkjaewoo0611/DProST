#!/bin/bash
project_root=$(pwd)
dataset_root=/bjw0611/6D-Pose/Dataset  # change path to what you want
mkdir -p ${dataset_root}
cd ${dataset_root}
SRC=https://bop.felk.cvut.cz/media/data/bop_datasets

# LINEMOD dataset
gdown https://drive.google.com/uc?id=1a_-_l_BYxuIB3cFewafvbwCgi7EbtrXD  --no-check-certificate           # test_bbox, index, preprocessed files
wget $SRC/lm_models.zip  --no-check-certificate                                                           # models
wget $SRC/lm_test_all.zip --no-check-certificate                                                          # test, train
wget $SRC/lm_train_pbr.zip --no-check-certificate                                                         # pbr

unzip LINEMOD.zip
rm LINEMOD.zip
unzip lm_models.zip -d LINEMOD
rm -rf LINEMOD/models_eval
rm lm_models.zip
unzip lm_test_all.zip -d LINEMOD
rm lm_test_all.zip
unzip lm_train_pbr.zip -d LINEMOD
rm lm_train_pbr.zip

cd LINEMOD
ln -s test train       # train and test are both from real dataset
mv train_pbr pbr
cd ..

### optional (only for real + syn data experiment)
### wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar     # background
### tar -xvf VOCtrainval_11-May-2012.tar
### mv VOCdevkit/VOC2012/JPEGImages background
### rm -rf VOCdevkit
### rm VOCtrainval_11-May-2012.tar
### download LINEMOD_6D/LM6d_converted/LM6d_refine_syn/data/rendered in  https://github.com/liyi14/mx-DeepIM/blob/master/prepare_data.md  # syn


# OCCLUSION dataset
gdown https://drive.google.com/uc?id=1wxc-ZR5tFBaqf1LmCHxNwOrZraWWIZcK  --no-check-certificate           # test_bbox, index, preprocessed files
wget $SRC/lmo_models.zip --no-check-certificate                                                            # models
wget $SRC/lmo_test_all.zip --no-check-certificate                                                          # test, train

unzip OCCLUSION.zip
rm OCCLUSION.zip
rm -rf OCCLUSION/models_eval
unzip lmo_models.zip -d OCCLUSION
rm lmo_models.zip
unzip lmo_test_all.zip -d OCCLUSION
rm lmo_test_all.zip

cd OCCLUSION
ln -s ../LINEMOD/test train       # shares training set with LM
ln -s ../LINEMOD/pbr pbr           # shares pbr set with LM
cd ..


# YCBV Dataset
gdown https://drive.google.com/uc?id=1NSGv7Hoj7cSKtlHljmznzd1vs6xo4Z0w  --no-check-certificate             # test_bbox, index, preprocessed files
wget $SRC/ycbv_models.zip --no-check-certificate                                                            # models
wget $SRC/ycbv_test_all.zip --no-check-certificate                                                          # test
wget $SRC/ycbv_train_pbr.zip --no-check-certificate                                                         # pbr
wget $SRC/ycbv_train_real.zip --no-check-certificate                                                        # train

unzip YCBV.zip
rm YCBV.zip
unzip ycbv_models.zip -d YCBV
rm -rf YCBV/models_eval
rm ycbv_models.zip
unzip ycbv_test_all.zip -d YCBV
rm ycbv_test_all.zip
unzip ycbv_train_pbr.zip -d YCBV
rm ycbv_train_pbr.zip
unzip ycbv_train_real.zip -d YCBV
rm ycbv_train_real.zip










cd YCBV
mv train_pbr pbr
mv train_real train       
cd ..

cd ${project_root}
ln -s ${dataset_root} Dataset
