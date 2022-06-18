#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/90deg1000sample \
    --img_size 112 \
    --z_dim 100 \
    --conv_unit_ch 64 \
    --load_weights_dir ../../weights/1000sample112pixel100z64ch00005lrd00005lrg00005lre100batch100epoch \
    --flag_show_reconstracted_images

cd $original_dir