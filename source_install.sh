#!/bin/bash
conda update -n base -c defaults conda -y
conda create --name DProST python=3.7.0 -y
source activate DProST
conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install jupyter -y
conda install pytorch3d==0.5.0 -c pytorch3d -y
pip install pandas transforms3d pytz scipy scikit-image matplotlib imageio pypng plotly opencv-python Pillow transform3d tensorboardX tensorboard vispy PyOpenGL flatten-dict simple-gpu-scheduler gdown

cd utils
git clone https://github.com/thodan/bop_toolkit.git
cd ..