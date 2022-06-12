#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 train.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/5deg1000sample \
    --z_dim 100 \
    --img_size 112 \
    --batch_size 100 \
    --load_weights_dir ../../weights/1000sample112pixel100z1e-05lrd5e-05lrg5e-05lre100batch100epoch \
    --lr_dis 1e-6 \
    --lr_gen 1e-5 \
    --lr_enc 1e-5 \
    --num_epochs 100

cd $original_dir