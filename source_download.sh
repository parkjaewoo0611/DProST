#!/bin/bash
project_root=$(pwd)
dataset_root=/home/jw/E/Instance-Pose/Dataset  # change path to what you want
mkdir -p ${dataset_root}
cd ${dataset_root}
SRC=https://bop.felk.cvut.cz/media/data/bop_datasets

# LINEMOD dataset
gdown https://drive.google.com/uc?id=1zjihT3BkgkEQ781Ns7ZQrYsksXSl-i-D             # test_bbox, index, preprocessed files
wget $SRC/lm_models.zip  --no-check-certificate                                                           # models
wget $SRC/lm_test_all.zip --no-check-certificate                                                          # test, train
wget $SRC/lm_train_pbr.zip --no-check-certificate                                                         # pbr

unzip LINEMOD.zip
unzip lm_models.zip -d LINEMOD
unzip lm_test_all.zip -d LINEMOD
unzip lm_train_pbr.zip -d LINEMOD

cd LINEMOD
ln -s test train       # train and test are both from real dataset
mv train_pbr pbr
cd ..

rm LINEMOD.zip
rm -rf LINEMOD/models_eval
rm lm_models.zip
rm lm_test_all.zip
rm lm_train_pbr.zip

### optional (only for real + syn data experiment)
### wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar     # background
### tar -xvf VOCtrainval_11-May-2012.tar
### mv VOCdevkit/VOC2012/JPEGImages background
### rm -rf VOCdevkit
### rm VOCtrainval_11-May-2012.tar
### download LINEMOD_6D/LM6d_converted/LM6d_refine_syn/data/rendered in  https://github.com/liyi14/mx-DeepIM/blob/master/prepare_data.md  # syn


# OCCLUSION dataset
gdown https://drive.google.com/uc?id=1HyZOfDNFcliRG270-lEGRF_NeUSPQLWu             # test_bbox, index, preprocessed files
wget $SRC/lmo_models.zip --no-check-certificate                                                            # models
wget $SRC/lmo_test_all.zip --no-check-certificate                                                          # test, train

unzip OCCLUSION.zip
unzip lmo_models.zip -d OCCLUSION
unzip lmo_test_all.zip -d OCCLUSION

cd OCCLUSION
ln -s ../LINEMOD/test train       # shares training set with LM
ln -s ../LINEMOD/pbr pbr           # shares pbr set with LM
cd ..

rm OCCLUSION.zip
rm -rf OCCLUSION/models_eval
rm lmo_models.zip
rm lmo_test_all.zip


# YCBV Dataset
gdown https://drive.google.com/uc?id=1jdl9oU9g_TxK3xO9gRavWHDT7359TlLy               # test_bbox, index, preprocessed files
wget $SRC/ycbv_models.zip --no-check-certificate                                                            # models
wget $SRC/ycbv_test_all.zip --no-check-certificate                                                          # test
wget $SRC/ycbv_train_pbr.zip --no-check-certificate                                                         # pbr
wget $SRC/ycbv_train_real.zip --no-check-certificate                                                        # train

unzip YCBV.zip
unzip ycbv_models.zip -d YCBV
unzip ycbv_test_all.zip -d YCBV
unzip ycbv_train_pbr.zip -d YCBV
unzip ycbv_train_real.zip -d YCBV

cd YCBV
mv train_pbr pbr
mv train_real train       
cd ..

rm YCBV.zip
rm -rf YCBV/models_eval
rm ycbv_models.zip
rm ycbv_test_all.zip
rm ycbv_train_pbr.zip
rm ycbv_train_real.zip

cd ${project_root}
ln -s ${dataset_root} Dataset
