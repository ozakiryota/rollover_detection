#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 train_with_l1loss.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/5deg10000sample \
    --img_size 112 \
    --z_dim 100 \
    --conv_unit_ch 64 \
    --batch_size 200 \
    --lr_dis 1e-4 \
    --lr_gen 1e-4 \
    --lr_enc 1e-4 \
    --num_epochs 100 \
    --l1_loss_weight 0.1

cd $original_dir