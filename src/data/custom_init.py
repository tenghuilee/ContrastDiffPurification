from torch.utils.data import Subset
import torchvision.datasets
import numpy as np

from .gtsrb import GTSRBDatasetTest

################################################################################
# MNIST
###############################################################################

def mnist_dataset_sub(root, transform=None, num_sub=-1, data_seed=0):
    val_data = torchvision.datasets.MNIST(root=root, transform=transform, download=True, train=False)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(val_data), num_sub, replace=False)
        val_data = Subset(val_data, partition_idx)

    return val_data

################################################################################
# GTSRB
###############################################################################

def gtsrb_dataset_sub(root, transform=None, num_sub=-1, data_seed=0, enable_crop=True):
    val_data = GTSRBDatasetTest(root=root, transform=transform, enable_crop=enable_crop)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(val_data), num_sub, replace=False)
        val_data = Subset(val_data, partition_idx)

    return val_data



################################################################################
# CIFAR100
###############################################################################
def cifar100_dataset_sub(root, transform=None, num_sub=-1, data_seed=0, enable_crop=True):
    val_data = torchvision.datasets.CIFAR100(root=root, transform=transform, download=True, train=False)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(val_data), num_sub, replace=False)
        val_data = Subset(val_data, partition_idx)

    return val_data

