from random import shuffle

import torch
import numpy as np

from typing import Tuple

import torchvision as vsn
from torch.utils.data import Dataset, DataLoader

from transforms import tforms

class SupervisedDataset(Dataset):

    def __init__(
        self,
        labeled_data: vsn.datasets = None,
        valid: bool = False,
    ):
        self.labeled_data = labeled_data
        self.mode = 'valid' if valid else 'train'

    def __getitem__(self, idx: int) -> dict:
        np.random.seed()
        # load an augment the image
        orig_img, targ = self.labeled_data[idx]
        img = tforms[self.mode](orig_img)

        return {
            'img': img,
            'targ': torch.tensor(targ),
        }

    def __len__(self):
        return len(self.labeled_data)

class SemiSupervisedDataset(Dataset):

    def __init__(
        self,
        dataset: vsn.datasets = None,
        valid: bool = False,
        balance_rate: float = 0.25,
        semi_supervised: bool = True
    ):
        self.dataset = dataset
        self.mode = 'valid' if valid else 'train'
        self.balance_rate = balance_rate
        self.semi_supervised = semi_supervised

    def __getitem__(self, idx: int) -> dict:

        if self.semi_supervised:
            if np.random.random() > self.balance_rate:
                # choose an unlabeled image
                indices = np.where(self.dataset.targets == -1)[0]
            else:
                # choose a labeled image
                indices = np.where(self.dataset.targets != -1)[0]

            idx = np.random.choice(indices)

        # load the data
        orig_img, targ = self.dataset[idx]
        # augment image twice
        img = tforms[self.mode](orig_img)
        ema_img = tforms[self.mode](orig_img)

        return {
            'img': img,
            'ema_img': ema_img,
            'targ': torch.tensor(targ),
            'img_idx': idx
        }

    def __len__(self):
        return len(self.dataset)

def balanced_class_split(labeld_targs: np.array) -> Tuple[np.array, np.array]:
    labels = np.array(labeld_targs)

    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))

    for class_idx in range(10):
        idx = np.where(labels == class_idx)[0]
        idx = np.random.choice(idx, 400, replace=False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx, dtype=np.int32)

    # make sure there is no overlap in supervised and unsupervised
    unlabeled_idx = np.array([x for x in unlabeled_idx if x not in labeled_idx], dtype=np.int32)

    print(len(unlabeled_idx), len(labeled_idx))

    return labeled_idx, unlabeled_idx

def get_data_loaders(batch_size: int = 32, num_labeled: int = 50000):
    # get the labeled data
    labeled_trainset = vsn.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=None
    )

    # get the validation dataset
    validset = vsn.datasets.CIFAR10(
        root='../data', train=True,
        download=True, transform=None
    )

    # ensure no overlap in train and valid
    valid_idx = np.random.randint(len(labeled_trainset), size=int(len(labeled_trainset) * 0.1))
    train_idx = [idx for idx in range(len(labeled_trainset)) if idx not in valid_idx]

    validset.data = np.array(validset.data)[valid_idx]
    validset.targets = np.array(validset.targets)[valid_idx]

    labeled_trainset.data = np.array(labeled_trainset.data)[train_idx]
    labeled_trainset.targets = np.array(labeled_trainset.targets)[train_idx]

    semi_supervised = num_labeled < len(labeled_trainset)

    if semi_supervised:

        _, unlabeled_idx = balanced_class_split(labeled_trainset.targets)
        labeled_trainset.targets[unlabeled_idx] = -1

    train_dataset = SemiSupervisedDataset(
        dataset = labeled_trainset,
        semi_supervised = semi_supervised
    )

    valid_dataset = SupervisedDataset(
        labeled_data=validset,
        valid=True
    )

    # set up the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # get the testset
    testset = vsn.datasets.CIFAR10(
        root='./data', train=False,
        download=True,
        transform=tforms['valid']
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )

    return train_loader, valid_loader, test_loader