#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/90deg1000sample \
    --img_size 64 \
    --z_dim 128 \
    --model_name dcgan \
    --conv_unit_ch 64 \
    --load_weights_dir ../../weights/dcgan64pixel128z64ch00001lrd00001lrg00001lre1000sample128batch100epoch \
    --flag_show_reconstracted_images

cd $original_dir