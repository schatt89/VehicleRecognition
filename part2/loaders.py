import torch
import torch.utils.data
from torchvision import datasets

from utils import plot_images

from samplers import valid_and_train_samplers
from dataset import TestImageFolder


def get_train_valid_loader(data_dir: str,
                           batch_size: int,
                           data_transforms: dict,
                           random_state: int,
                           weighted_sampler: bool,
                           valid_size: float,
                           shuffle: bool,
                           show_sample: bool,
                           num_workers: int,
                           pin_memory: bool):
    """
    Utility function for loading and returning train and valid.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - data_transforms: whether to apply the data augmentation scheme
    - random_state: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 3x8 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - weighted: whether to use a weighted sampler
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    # load the dataset
    train_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    valid_dataset = datasets.ImageFolder(data_dir, data_transforms['valid'])

    train_sampler, valid_sampler, cls_to_weight = valid_and_train_samplers(
        train_dataset, weighted_sampler, valid_size, random_state)

    train_dataset.cls_to_weight = cls_to_weight

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, 
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, 
        pin_memory=pin_memory,
    )

    print(f'Total: {len(train_dataset)}; Train/Valid: {len(train_sampler)}/{len(valid_sampler)}')

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=5*8, sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, data_dir, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir: str,
                    batch_size: int,
                    data_transforms: dict,
                    num_workers: int,
                    pin_memory: bool):
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
