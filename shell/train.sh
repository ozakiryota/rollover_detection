#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 train.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/5deg \
    --z_dim 100 \
    --lr_dis 1e-6 \
    --lr_gen 1e-4 \
    --lr_enc 1e-4 \
    --num_epochs 10

cd $original_dir