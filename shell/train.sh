#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 train.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/5deg1000sample \
    --img_size 112 \
    --z_dim 100 \
    --conv_unit_ch 64 \
    --batch_size 100 \
    --load_weights_dir ../../weights/1000sample112pixel100z1e05lrd1e05lrg1e05lre100batch100epoch \
    --lr_dis 1e-5 \
    --lr_gen 1e-5 \
    --lr_enc 1e-5 \
    --num_epochs 100

cd $original_dir