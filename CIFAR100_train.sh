#! /bin/bash

ROOT_DIR=$PWD

# pip install lightning

# python CIFAR100_train.py \
#     --backbone wrideresnet-28-10 \
#     --data_root "./DiffPure/dataset/cifar100"\
#     --ckpt_root "./cifar100_checkpoints_round_2" \
#     --resume_ckpt "./cifar100_checkpoints_round_2/model-epoch=82-val_loss=1.52.ckpt" 

# python CIFAR100_train.py \
#     --backbone wrideresnet-28-10 \
#     --data_root "./DiffPure/dataset/gtsrb"\
#     --ckpt_root "./gtsrb_wrn-28-10_checkpoint" \
#     --dataset "gtsrb" \
#     --dataset_dir "./DiffPure/dataset/gtsrb" \
#     --train_max_epochs 256 \
#     --checkpoint "./gtsrb_model_final_wrn-28-10.ckpt" \

# python CIFAR100_train.py \
#     --backbone wrideresnet-70-16 \
#     --data_root "./DiffPure/dataset/cifar100"\
#     --ckpt_root "./cifar100-70-16_checkpoint" \
#     --dataset "cifar100" \
#     --dataset_dir "./DiffPure/dataset/cifar100" \
#     --train_max_epochs 256 \
#     --checkpoint "./cifar100_model_final_wrn-70-16.ckpt" \
#     --train_batch_size 64 \
#     --val_batch_size 64 \

python CIFAR100_train.py \
    --backbone "wrideresnet-28-10" \
    --data_root "./DiffPure/dataset/cifar100"\
    --ckpt_root "./cifar100-28-10_checkpoint" \
    --dataset "cifar100" \
    --dataset_dir "./DiffPure/dataset/cifar100" \
    --train_max_epochs 1024 \
    --checkpoint "./cifar100_model_final_wrn-28-10.ckpt" \
    --resume_ckpt "./cifar100-28-10_checkpoint/model-epoch=168-val_loss=1.74.ckpt" \
    --train_batch_size 64 \
    --val_batch_size 64 \

# python CIFAR100_train.py \
#     --backbone resnet18 \
#     --data_root "./DiffPure/dataset/gtsrb"\
#     --ckpt_root "./gtsrb_resnet18_checkpoint" \
#     --dataset "gtsrb" \
#     --dataset_dir "./DiffPure/dataset/gtsrb" \
#     --train_max_epochs 128 \
#     --checkpoint "./gtsrb_model_final_resnet18.ckpt"


