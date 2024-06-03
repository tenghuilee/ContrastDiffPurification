#!/usr/bin/env bash
cuda_index=$1
SEED1=$2
SEED2=$3

for t in 75; do
  for adv_eps in 0.5; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=$cuda_index python eval_sde_adv_contrastive.py --exp ./exp_results_contrastive_wideresnet-28-10 --config cifar10.yml \
          -i cifar10-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-L2-eot20 \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 64 \
          --num_sub 64 \
          --domain cifar10 \
          --classifier_name cifar10-wideresnet-28-10 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \
          --attack_version rand \
          --eot_iter 20 \
          --lp_norm L2 \
          --contrastiveive_classifier cifar10-wideresnet-28-10 \
          --constructive_n_classes 10 \
          --constructive_optim_lr 0.01 \
          --constructive_drift_min 0 \
          --contrastive_delta_grad_mul 500 \
          --contrastive_temperature 1.0 \
          --constructive_drift_max 100000 \

      done
    done
  done
done
