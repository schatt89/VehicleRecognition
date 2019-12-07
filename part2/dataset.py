import os
from glob import glob
from typing import Dict, Tuple, List, Iterator

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, WeightedRandomSampler, SequentialSampler
from torchvision import datasets, transforms

from utils import plot_images

# class ImbalancedDatasetSampler(Sampler):

#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Arguments:
#         indices (list, optional): a list of indices
#         num_samples (int, optional): number of samples to draw
    
#     Reference: 
#     github.com/ufoym/imbalanced-dataset-sampler/blob/a14e7cbf3a712f280fc02615c6b690bd51a3a8a8/sampler.py
#     """

#     def __init__(self, dataset, indices=None, num_samples=None):

#         # if indices is not provided,
#         # all elements in the dataset will be considered
#         self.indices = list(range(len(dataset))) \
#             if indices is None else indices

#         # if num_samples is not provided,
#         # draw `len(indices)` samples in each iteration
#         self.num_samples = len(self.indices) \
#             if num_samples is None else num_samples

#         # distribution of classes in the dataset
#         label_to_count = {}
#         for idx in self.indices:
#             label = self._get_label(dataset, idx)
#             if label in label_to_count:
#                 label_to_count[label] += 1
#             else:
#                 label_to_count[label] = 1

#         # weight for each sample
#         weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
#                    for idx in self.indices]
#         self.weights = torch.DoubleTensor(weights)

#     def _get_label(self, dataset, idx):
#         dataset_type = type(dataset)
#         if dataset_type is torchvision.datasets.ImageFolder:
#             return dataset.imgs[idx][1]
#         else:
#             raise NotImplementedError

#     def __iter__(self):
#         return (self.indices[i] for i in torch.multinomial(
#             self.weights, self.num_samples, replacement=True))

#     def __len__(self):
#         return self.num_samples



def sklearn_stratified_train_valid_splits(
        y: np.ndarray,
        valid_size: float,
        random_state: int
    ) -> Iterator[Tuple[list, list]]:

    # X which is not used in .split anyway
    X = np.arange(len(y)).reshape(-1, 1)
    # n_splits=int(1/0.25): 4
    skf = StratifiedKFold(n_splits=int(1/valid_size), shuffle=True, random_state=random_state)
    return skf.split(X, y)

def calculate_samples_weights(
        train_index: list,
        path_cls: List[Tuple[str, int]]
    ) -> torch.DoubleTensor:

    # filter path_cls for training dataset
    train_path_cls = [path_cls[idx] for idx in train_index]
    train_sample_cls = [Class for Path, Class in train_path_cls]
    # count each class only in training; if valid is used -> may overfit
    cls_to_count = {cls: train_sample_cls.count(cls) for cls in set(train_sample_cls)}
    # weights are disproportional to the number of occurences for each class in train dataset
    cls_to_weight = {cls: 1 / cls_to_count[cls] for cls, count in cls_to_count.items()}
    # initialize the weights with zeros
    samples_weights = torch.zeros(len(path_cls)).double()

    # fill all training indices with weights according to the sample size; valid idx have probs = 0
    for idx in train_index:
        sample_path, sample_class = path_cls[idx]
        samples_weights[idx] = cls_to_weight[sample_class]

    return samples_weights


def valid_and_weighted_train_samplers(
            train_dataset: Dataset,
            valid_size: float,
            random_state: int,
    ) -> Tuple[Sampler, Sampler]:
    # path_cls a list of tuples: (path, class_idx)
    path_cls = train_dataset.imgs
    # extract only the classes and make them numpy array
    samples_classes = np.array([Class for Path, Class in path_cls])
    # train-valid split in a stratified manner (return splits iterator)
    skf_splits = sklearn_stratified_train_valid_splits(samples_classes, valid_size, random_state)
    # select only the first split (can be expanded to K-fold cross validation)
    train_index, valid_index = next(skf_splits)
    # weights for each sample in ImageDataset (train: weighted, valid: all zeros)
    samples_weights = calculate_samples_weights(train_index, path_cls)
    # sampled indices (which must belong only to train_index) <-
    train_sampler = WeightedRandomSampler(samples_weights, len(train_index))

    # cls_to_count_obs = {cls: 0 for cls in set(samples_classes)}
    # for i in range(100):
    #     for train_idx in train_sampler:
    #         cls_to_count_obs[samples_classes[train_idx]] += 1
    # print(cls_to_count_obs) # {0: 2501153, 1: 2497812, 2: 2501595, 3: 2499540}

    return train_sampler, SubsetRandomSampler(valid_index)



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
                           random_state: int,
                           weighted: bool,
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
    - random_state: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
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

    if weighted:
        print('Using weighted')
        train_sampler, valid_sampler = valid_and_weighted_train_samplers(
            train_dataset, valid_size, random_state)

    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_state)
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

    print(f'Total: {len(train_dataset)} images; Train/Valid: {train_sampler}/{len(valid_sampler)}')

    print(len(train_sampler), len(valid_sampler), len(train_dataset))
    print(len(train_loader), len(valid_loader))

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, sampler=train_sampler, num_workers=num_workers, 
            pin_memory=pin_memory,
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
