import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data.gtsrb import GTSRB

import robustbench.model_zoo.architectures.dm_wide_resnet as dm_wide_resnet
import robustbench.model_zoo.architectures.resnet as resnet

_args = argparse.ArgumentParser()
_args.add_argument("--backbone", type=str, default="wrideresnet-28-10")
_args.add_argument("--data_root", type=str, default="./src/dataset/cifar100",
                   help="the root path to dataset CIFAR100")
_args.add_argument("--checkpoint", type=str,
                   default="./cifar100_model_final_{backbone}.ckpt", help="The saved checkpoint")
_args.add_argument("--ckpt_root", type=str, default="./cifar100_checkpoints",
                   help="the root path to save checkpoints")
_args.add_argument("--resume_ckpt", type=str, default="last",
                   help="the checkpoint path to resume training")
_args.add_argument("--optim", type=str, default="adam", help="the optimizer")
_args.add_argument("--dataset", type=str, default="cifar100", help="dataset to train on")
_args.add_argument("--dataset_dir", type=str, default="./src/dataset/cifar100", help="directory to dataset")
_args.add_argument("--test_mode", action="store_true")
_args.add_argument("--train_max_epochs", type=int, default=512, help="max epochs for train")
_args.add_argument("--train_batch_size", type=int, default=1024, help="batch size for train")
_args.add_argument("--val_batch_size", type=int, default=1024, help="batch size for val")
args = _args.parse_args()
args.checkpoint = args.checkpoint.format(backbone=args.backbone)

print(args)


class DMClassifier(pl.LightningModule):
    def __init__(self, backbone, num_classes=100, optim=""):
        super(DMClassifier, self).__init__()
        if num_classes == 100:
            _mean = dm_wide_resnet.CIFAR100_MEAN
            _std = dm_wide_resnet.CIFAR100_STD
        elif num_classes == 10:
            _mean = dm_wide_resnet.CIFAR10_MEAN
            _std = dm_wide_resnet.CIFAR10_STD
        if backbone == "wrideresnet-28-10":
            self.resnet = dm_wide_resnet.DMWideResNet(
                num_classes=num_classes,
                depth=28,
                width=10,
                activation_fn=dm_wide_resnet.Swish,
                mean=_mean,
                std=_std,
            )
        elif backbone == "wrideresnet-70-16":
            self.resnet = dm_wide_resnet.DMWideResNet(
                num_classes=num_classes,
                depth=70,
                width=16,
                activation_fn=dm_wide_resnet.Swish,
                mean=_mean,
                std=_std,
            )
        elif backbone == "resnet18":
            self.resnet = resnet.ResNet(
                block=resnet.BasicBlock,
                num_blocks=[2, 2, 2, 2],
                num_classes=num_classes,
            )
        else:
            raise NotImplementedError(f"{backbone} is not supported")

        self.resnet.train()
        self.criterion = nn.CrossEntropyLoss()
        self.__optim = optim.lower()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        predict_labels = outputs.argmax(dim=1)
        acc = (predict_labels == labels).sum().item() / labels.size(0)
        self.log('test_acc', acc)
        return acc

    def configure_optimizers(self):
        if self.__optim == 'sgd':
            optimizer = optim.SGD(self.trainer.model.parameters(), lr=5e-4)
        elif self.__optim == 'adam':
            optimizer = optim.Adam(self.trainer.model.parameters(), lr=2e-4)
        return optimizer


if args.test_mode:
    # Data
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
else:
    # Data
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=(32, 32)),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    if args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == 'gtsrb':
        train_dataset = GTSRB(
            root=args.data_root,
            train=True,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=8)

num_classes = 100
if args.dataset == 'cifar100':
    num_classes = 100
    val_dataset = datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=transform)
elif args.dataset == 'gtsrb':
    num_classes = 43
    val_dataset = GTSRB(
        root=args.data_root,
        train=False,
        transform=transform,
    )
else:
    raise NotImplementedError(f"dataset {args.dataset} is not supported")

val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                        shuffle=False, num_workers=8)

os.makedirs(args.ckpt_root, exist_ok=True)

trainer = pl.Trainer(
    max_epochs=args.train_max_epochs,
    # enable_checkpointing=False,
    enable_progress_bar=True,
    log_every_n_steps=1,
    default_root_dir=args.ckpt_root,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.ckpt_root,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        ModelCheckpoint(
            dirpath=args.ckpt_root,
            filename='model-latest',
        ),
    ],
)

if args.test_mode:
    # Test
    try:
        model = DMClassifier.load_from_checkpoint(
            args.checkpoint,
            backbone=args.backbone,
            num_classes=num_classes,
        )
    except Exception:
        model = DMClassifier(args.backbone, num_classes=num_classes)
        model.resnet.load_state_dict(torch.load(args.checkpoint))

    trainer.test(model, val_loader)
else:
    # Training
    model = DMClassifier(args.backbone, num_classes=num_classes, optim=args.optim)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
    trainer.save_checkpoint(args.checkpoint)
