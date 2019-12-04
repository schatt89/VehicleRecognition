import os
from glob import glob

import torch
import numpy as np

from PIL import Image

from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset


class TestImageFolder(Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.dataset = sorted(glob(os.path.join(data_dir, '*.jpg')))
        self.transforms = transforms
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset[idx]

        pil_image = self.pil_loader(path)
        if self.transforms is not None:
            pil_image = self.transforms(pil_image)

        return pil_image, path

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def get_train_valid_loader(data_dir,
                           batch_size,
                           data_transforms,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - data_transforms: whether to apply the data augmentation scheme
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """


    # load the dataset
    train_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    valid_dataset = datasets.ImageFolder(data_dir, data_transforms['valid'])

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, data_dir, labels)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    data_transforms,
                    num_workers=4,
                    pin_memory=False):
    """
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - data_transforms: whether to apply the data augmentation scheme
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - test_loader: test set iterator.
    """

    test_dataset = TestImageFolder(data_dir, data_transforms['valid'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory=pin_memory,
    )
    return test_loader
