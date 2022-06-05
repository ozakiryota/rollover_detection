#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/90deg \
    --img_size 64 \
    --z_dim 100 \
    --weights_dir ../../weights/1000sample64pixel100z1e-05lrd0.0001lrg0.0001lre100batch100epoch \
    --flag_show_reconstracted_images

cd $original_dir