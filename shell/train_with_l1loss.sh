#!/bin/bash

original_dir=$(pwd)
pyscr_dir=../pyscr/exec

cd $pyscr_dir

python3 train_with_l1loss.py \
    --dataset_dirs $HOME/dataset/rollover_detection/airsim/5deg1000sample \
    --img_size 64 \
    --z_dim 128 \
    --model_name dcgan \
    --conv_unit_ch 64 \
    --batch_size 128 \
    --lr_dis 1e-4 \
    --lr_gen 1e-4 \
    --lr_enc 1e-4 \
    --num_epochs 100 \
    --save_weights_step 10 \
    --l1_loss_weight 1.0

cd $original_dir