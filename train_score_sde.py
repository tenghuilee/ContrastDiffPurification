from dataclasses import dataclass, field
from typing import Any
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils

import score_sde.losses as losses
import score_sde.sampling as sampling
from score_sde import sde_lib
from score_sde.models import utils as mutils
from score_sde.models.ema import ExponentialMovingAverage
import argparse

from src.data.gtsrb import GTSRB

from basic_config_setting import *


def get_data_scaler(is_data_centered: bool):
    """Data normalizer. Assume data are always in [0, 1]."""
    if is_data_centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(is_data_centered: bool):
    """Inverse data normalizer."""
    if is_data_centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x




class ScoreSDETrain(pl.LightningModule):
    def __init__(self, config: ConfigMain):
        super().__init__()
        self.config = config
        self.model = mutils.create_model(config) # type: nn.Module

        sde, sampling_eps = self.build_sde()

        self.sampling_eps = sampling_eps

        self.loss_func = self.get_sde_loss_func(sde)

    
    def build_sde(self):
        config = self.config
        sde_type = config.training.sde.lower()

        # Setup SDEs
        if sde_type == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                                beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif sde_type == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                                   beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif sde_type == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                                sigma_max=config.model.sigma_max, N=config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        return sde, sampling_eps
    
    def get_sde_loss_func(self, sde):
        continuous = self.config.training.continuous
        reduce_mean = self.config.training.reduce_mean
        likelihood_weighting = self.config.training.likelihood_weighting
        if continuous:
            loss_fn_train = losses.get_sde_loss_fn(sde, True, reduce_mean=reduce_mean,
                                                   continuous=True, likelihood_weighting=likelihood_weighting)
        else:
            assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                loss_fn_train = losses.get_smld_loss_fn(
                    sde, True, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                loss_fn_train = losses.get_ddpm_loss_fn(
                    sde, True, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended.")
        
        return loss_fn_train

    def training_step(self, batch, batch_idx):
        images, labels = batch

        loss = self.loss_func(self.model, images)

        self.log('train_loss', loss, prog_bar=True)

        return loss
    
    # def test_step(self, batch, batch_idx):
    #     images, labels = batch
    #     pass


    def configure_optimizers(self):
        return losses.get_optimizer(self.config, self.trainer.model.parameters())

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        warmup = self.config.optim.warmup
        steps = self.trainer.global_step
        if warmup > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = self.config.optim.lr * \
                    np.minimum(steps / warmup, 1.0)
        optimizer.step(closure=optimizer_closure)
    
    @torch.no_grad()
    def get_sample_func(self):
        # self.ema

        sde, sampling_eps = self.build_sde()
        inverse_scaler = get_data_inverse_scaler(self.config.data.centered)
        _config = self.config
        device = self.device

        def __inner(batch_size, model):

            sampling_shape = (batch_size, _config.data.num_channels,
                                _config.data.image_size, _config.data.image_size)

            sampler_name = _config.sampling.method

            # copy from score_sde
            # Probability flow ODE sampling with black-box ODE solvers
            if sampler_name.lower() == 'ode':
                sampling_fn = sampling.get_ode_sampler(sde=sde,
                                            shape=sampling_shape,
                                            inverse_scaler=inverse_scaler,
                                            denoise=_config.sampling.noise_removal,
                                            eps=sampling_eps,
                                            device=device)
            # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
            elif sampler_name.lower() == 'pc':
                predictor = sampling.get_predictor(_config.sampling.predictor.lower())
                corrector = sampling.get_corrector(_config.sampling.corrector.lower())
                sampling_fn = sampling.get_pc_sampler(sde=sde,
                                            shape=sampling_shape,
                                            predictor=predictor,
                                            corrector=corrector,
                                            inverse_scaler=inverse_scaler,
                                            snr=_config.sampling.snr,
                                            n_steps=_config.sampling.n_steps_each,
                                            probability_flow=_config.sampling.probability_flow,
                                            continuous=_config.training.continuous,
                                            denoise=_config.sampling.noise_removal,
                                            eps=sampling_eps,
                                            device=device)
            else:
                raise ValueError(f"Sampler name {sampler_name} unknown.")

            return sampling_fn(model)

        return __inner
    
    @torch.no_grad()
    def generate_sample(self, batch_size: int, sample_func: callable = None):
        if sample_func is None:
            sample_func = self.get_sample_func()

        # ema = ExponentialMovingAverage(self.model.parameters(), decay=config.model.ema_rate) 
        # ema.store(self.model.parameters())
        # ema.copy_to(self.model.parameters())

        sample, n = sample_func(batch_size, self.model)

        # ema.restore(self.model.parameters())
        return sample


_args = argparse.ArgumentParser()
_args.add_argument("--mode", type=str, default="train", help="train, test, generate")
_args.add_argument("--checkpoint", type=str, default="./cifar100_ncsnpp_scoresde_model.ckpt", help="checkpoint path")
_args.add_argument("--resume_checkpoint", type=str, default=None, help="checkpoint path for continuous train")
_args.add_argument("--generate_batch_size", type=int, default=64, help="batch size for generate")
_args.add_argument("--generate_file", type=str, default="./generate/samples.png", help="directory to save generate samples")
_args.add_argument("--config_path", type=str,
                   default="./DiffPure/configs/cifar100.yml", help="config file path")
_args.add_argument("--ckpt_root", type=str, default=None, help="directory to save checkpoints")
_args.add_argument("--dataset", type=str, default="cifar100", help="dataset to train on")
_args.add_argument("--dataset_dir", type=str, default="./DiffPure/dataset/cifar100", help="directory to dataset")
_args.add_argument("--train_max_epochs", type=int, default=1600, help="max epochs for train")
args = _args.parse_args()
main_config = ConfigMain.from_yaml(args.config_path)

if args.mode == "generate":
    print("generate the sample images using pretrained checkpoint", args.checkpoint)

    try:
        model = ScoreSDETrain.load_from_checkpoint(args.checkpoint, config=main_config).cuda()
    except Exception as e:
        model = ScoreSDETrain(main_config)
        model.model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
        model = model.cuda()

    model.eval()

    samples = model.generate_sample(args.generate_batch_size, model.get_sample_func())

    # save with torch vision
    torchvision.utils.save_image(samples, args.generate_file)
    print("samples saved to", args.generate_file)

elif args.mode == "train":

    _transfrom_pip = [
            transforms.Resize((main_config.data.image_size, main_config.data.image_size)),
            # transforms.RandomHorizontalFlip(), # do not random Horizontal Flip
            transforms.ToTensor(),
    ]

    if main_config.data.centered:
        _transfrom_pip.append(get_data_scaler(True))

    if args.dataset == "cifar100":
        dataset = datasets.CIFAR100(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transforms.Compose(_transfrom_pip),
        )
    elif args.dataset == "gtsrb":
        dataset = GTSRB(
            root=args.dataset_dir,
            train=True,
            transform=transforms.Compose(_transfrom_pip),
            enable_crop=False,
        )
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported")

    train_dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    if args.ckpt_root is not None:
        os.makedirs(args.ckpt_root, exist_ok=True)

    # configure train
    trainer = pl.Trainer(
        max_epochs=args.train_max_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir=args.ckpt_root,
    )

    model = ScoreSDETrain(main_config)
    trainer.fit(model, train_dataloaders=train_dataloader,
                ckpt_path=args.resume_checkpoint)
    
    trainer.save_checkpoint(args.checkpoint)
else:
    raise ValueError(f"mode ({args.mode}) must be 'train', 'test' or 'generate'")
