#! /bin/bash

# pip install lightning

python train_score_sde.py \
    --mode "train" \
    --ckpt_root "./gtsrb_ncsnpp" \
    --dataset "gtsrb" \
    --dataset_dir "./src/dataset/gtsrb" \
    --checkpoint "./gtsrb_ncsnpp_score_sde.ckpt" \
    --config_path "./src/configs/gtsrb.yml" \
    --train_max_epochs 1800 \
    --resume_checkpoint "./gtsrb_ncsnpp_score_sde.ckpt" \


