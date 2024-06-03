#! /bin/bash


ROOT_DIR=$PWD

# pip install lightning

# python train_score_sde.py --mode "train" --train_restore_checkpoint "./cifar_ncsnpp_continuous_epoch=1599-step=78400.ckpt"

# python train_score_sde.py --mode "train" --ckpt_root "./cifar100_ncsnpp"

python train_guided_diffusion.py \
    --mode train \
    --train_devices "auto" \
    --class_conditional False \




