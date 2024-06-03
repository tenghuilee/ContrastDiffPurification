import argparse
import yaml
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.strategies as pl_strategies
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

from src.guided_diffusion import resample, script_util

# torch.set_float32_matmul_precision('medium' | 'high')
torch.set_float32_matmul_precision('high')

class GuidedDiffusionTrainer(pl.LightningModule):

    def __init__(
        self,
        model: script_util.UNetModel,
        diffusion: script_util.SpacedDiffusion,
        lr,
        ema_rate,
    ):
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.lr = lr
        self.ema_rate = ema_rate
        self.diffusion_schedule_sampler = resample.UniformSampler(diffusion)
        self.class_conditional = model.num_classes is not None
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        images, labels = batch 

        time_step, weights = self.diffusion_schedule_sampler.sample(
            images.shape[0],
            images.device,
        )
        model_kwargs = {}
        if self.class_conditional:
            model_kwargs["y"] = labels

        losses = self.diffusion.training_losses(
            self.model,
            images, # x_start
            time_step, # t
            model_kwargs=model_kwargs,
        )
        loss = (losses["loss"] * weights).mean()

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.trainer.model.parameters(), lr=self.lr)
    
def get_config_dict_cifar10(args):
    return {
        "image_size": 32,
        "class_cond": args.class_conditional,
        "learn_sigma": True,
        "num_channels": 128,
        "num_res_blocks": 3,
        "channel_mult": "1, 2, 2, 2",
        "num_heads": 2,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "attention_resolutions": "16, 8",
        "dropout": 0.3,
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "timestep_respacing": "",
        "use_kl": False,    
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_fp16": False,
        "use_new_attention_order": False,
    }


def builf_model_and_diffusion_cifar10(args, return_params=False):
    script_util.NUM_CLASSES = 10
    params = get_config_dict_cifar10(args)
    model, diffusion = script_util.create_model_and_diffusion(**params)
    if return_params:
        return model, diffusion, params
    else:
        return model, diffusion

class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, repeat_times: int, shuffle=False):
        super().__init__()
        self.dataset = dataset
        self.repeat_times = repeat_times
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_indexes = np.random.permutation(len(dataset))
        else:
            self.shuffle_indexes = None

    def __len__(self):
        return self.repeat_times * len(self.dataset)
    
    def __getitem__(self, index):
        real_idx = index // self.repeat_times
        if self.shuffle:
            real_idx = self.shuffle_indexes[real_idx]
        return self.dataset[real_idx]

def train_mode(args):
    model, diffusion = builf_model_and_diffusion_cifar10(args)

    # load the cifar10 train dataset
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    ckpt_tag = "class_cond" if args.class_conditional else "uncond"

    devices = args.train_devices

    try:
        devices =[int(_i) for _i in args.train_devices.split(",")] 
    except:
        pass

    trainer = pl.Trainer(
        devices=devices,
        max_epochs=args.epochs,
        # strategy=pl_strategies.DDPStrategy(process_group_backend="gloo"),
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=8,
        default_root_dir=args.root_dir,
        enable_model_summary=True,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath=args.root_dir,
                filename=ckpt_tag + '-cifar10-epoch{epoch:04d}-step{step:04d}',
                auto_insert_metric_name=False,
                every_n_train_steps=512,
                save_on_train_epoch_end=True,
            ),
        ]
    )

    trainer.fit(
        GuidedDiffusionTrainer(
            model=model,
            diffusion=diffusion,
            lr=args.lr,
            ema_rate=args.ema_rate,
        ),
        train_dataloaders=torch.utils.data.DataLoader(
            RepeatDataset(trainset, 64, shuffle=True),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        ),
        # ckpt_path="pretrain_checkpoints/guide_diffusion_cifar10_class/class_cond-cifar10-epoch0038-step60928.ckpt",
        ckpt_path="last",
    )

    trainer.save_checkpoint(args.checkpoint_path)

def eval_mode(args):
    model, diffusion = builf_model_and_diffusion_cifar10(args)

    device = torch.device("cuda:0")

    print("load from checkpoint")
    print(args.checkpoint_path)

    # load model
    model = GuidedDiffusionTrainer.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device,
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        ema_rate=args.ema_rate,
    ).eval().to(device)

    model_kwargs = {}
    if model.class_conditional:
        labels = torch.randint(0, 10, (args.eval_num_samples, ), dtype=torch.long).to(device)
        print(labels)
        label_text_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # map index back to text label
        max_txt_len = max([len(i) for i in label_text_map])
        # label_text = [label_text_map[i] for i in labels.data.cpu().tolist()]
        for _i, _idx in enumerate(labels.data.cpu().tolist()):
            print(label_text_map[_idx].rjust(max_txt_len), end=" ")
            if _i % 8 == 7:
                print("") # line break
        
        # set label for model kwargs
        model_kwargs["y"] = labels

    predict = diffusion.p_sample_loop(
        model.model,
        shape=(args.eval_num_samples, 3, 32, 32),
        model_kwargs=model_kwargs,
        device=device,
    )

    torchvision.utils.save_image(
        predict,
        args.eval_image_output,
    )

def checkpointing_mode(args):
    assert args.output_checkpoint_path is not None, "output checkpoint path is required"

    model, diffusion, params = builf_model_and_diffusion_cifar10(args, return_params=True)

    # load checkpoint 
    model = GuidedDiffusionTrainer.load_from_checkpoint(
        args.checkpoint_path,
        map_location=torch.device("cpu"),
        model=model,
        diffusion=diffusion,
        lr=args.lr,
        ema_rate=args.ema_rate,
    )

    # save checkpoint
    torch.save(model.model.state_dict(), args.output_checkpoint_path)
    with open(f"{args.output_checkpoint_path}.yaml", "w") as f:
        yaml.dump(params, f)

    print("checkpoint saved to {}".format(args.output_checkpoint_path))

def _arg_bool(x: str):
    return x.lower() in ("true", "1", "t", "yes", "y")


if __name__ == "__main__":
    _args = argparse.ArgumentParser()
    _args.add_argument("--mode", type=str, default="train")
    _args.add_argument("--batch_size", type=int, default=256)
    _args.add_argument("--lr", type=float, default=1e-4)
    _args.add_argument("--epochs", type=int, default=64)
    _args.add_argument("--ema_rate", type=float, default=0.995)
    _args.add_argument("--class_conditional", type=_arg_bool, default="true")
    _args.add_argument("--data_path", type=str, default="./DiffPure_2/dataset/")
    _args.add_argument("--root_dir", type=str, default="./pretrain_checkpoints/guide_diffusion_cifar10_class")
    _args.add_argument("--checkpoint_path", type=str, default="./pretrain_checkpoints/guide_diffusion_cifar10_class.ckpt")
    _args.add_argument("--output_checkpoint_path", type=str, default=None, help="output checkpoint path; only avaliable when --mode=checkpointing")
    _args.add_argument("--train_devices", type=str, default="auto")
    _args.add_argument("--eval_num_samples", type=int, default=32)
    _args.add_argument("--eval_image_output", type=str,
                       default="./pretrain_checkpoints/predicted_images.png")
    

    args = _args.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "checkpointing":
        checkpointing_mode(args)
    else:
        raise ValueError("mode must be one of [train, eval, checkpointing]")

