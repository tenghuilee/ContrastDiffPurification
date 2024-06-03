#!/usr/bin/env bash
cuda_index=$1
SEED1=$2
SEED2=$3

for t in 75; do
  for adv_eps in 0.5; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=$cuda_index python eval_sde_adv_contrastive.py --exp ./exp_results_gtsrb_contrastive_resnet18 --config gtsrb.yml \
          -i gtsrb-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-L2-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 64 \
          --num_sub 64 \
          --domain gtsrb \
          --classifier_name gtsrb-resnet18 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \
          --attack_version rand \
          --eot_iter 20 \
          --lp_norm L2 \
          --contrastiveive_classifier gtsrb-resnet18 \
          --constructive_n_classes 43 \
          --constructive_optim_lr 0.01 \
          --constructive_drift_min 40 \
          --constructive_drift_max 100000 \

      done
    done
  done
done
