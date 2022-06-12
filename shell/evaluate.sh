#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/90deg1000sample \
    --img_size 112 \
    --z_dim 100 \
    --load_weights_dir ../../weights/1000sample224pixel100z5e-05lrd5e-05lrg5e-05lre100batch100epoch \
    --flag_show_reconstracted_images

cd $original_dir