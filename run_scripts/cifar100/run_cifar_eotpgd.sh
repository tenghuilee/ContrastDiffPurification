#!/usr/bin/env bash
cuda_index=$1
SEED1=$2
SEED2=$3

for t in 100; do
  for adv_eps in 0.031373; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=$cuda_index python eval_sde_adv.py --exp ./exp_results_cifar100 --config cifar100.yml \
          -i cifar100-robust_adv-$t-eps$adv_eps-64x1-bm0-t0-end1e-5-cont-eot20-eotpgd \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 64 \
          --num_sub 64 \
          --domain cifar100 \
          --classifier_name cifar100-wideresnet-28-10 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \
          --attack_version eotpgd \
          --eot_iter 20

      done
    done
  done
done
