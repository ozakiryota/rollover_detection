#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/90deg \
    --weights_dir ../../weights/1000sample112pixel100z100batch200epoch \
    --flag_show_reconstracted_images

cd $original_dir