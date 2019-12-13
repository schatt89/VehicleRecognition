import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, WeightedRandomSampler
from typing import Dict, Tuple, Iterator, Iterable


def sklearn_stratified_train_valid_splits(
    targets: np.ndarray,
    valid_size: float,
    random_state: int
) -> Iterator[Tuple[list, list]]:

    # X which is not used in .split anyway
    X = np.arange(len(targets)).reshape(-1, 1)
    # n_splits=int(1/0.25): 4
    skf = StratifiedKFold(n_splits=int(1/valid_size), shuffle=True, random_state=random_state)
    return skf.split(X, targets)


def get_train_class_weights(targets: np.ndarray, train_index: Iterable) -> Dict[int, float]:
    # filter targets only those that belong to training set
    train_targets = [targets[idx] for idx in train_index]
    # count each class only in training; if valid is used -> may overfit
    cls_to_count = {cls: train_targets.count(cls) for cls in set(train_targets)}
    # weights are disproportional to the number of occurences for each class in train dataset
    cls_to_weight = {cls: 1 / cls_to_count[cls] for cls, count in cls_to_count.items()}
    return cls_to_weight


def calculate_samples_weights(
    train_index: list,
    targets: np.ndarray
) -> Tuple[torch.DoubleTensor, Dict[int, float]]:

    # dict: idx -> weight
    cls_to_weight = get_train_class_weights(targets, train_index)
    # initialize the weights with zeros
    samples_weights = torch.zeros(len(targets)).double()
    # fill all training indices with weights according to the sample size; valid idx have probs = 0
    for idx in train_index:
        sample_class = targets[idx]
        samples_weights[idx] = cls_to_weight[sample_class]

    return samples_weights, cls_to_weight

def valid_and_train_samplers(
    train_dataset: Dataset,
    weighted: bool,
    valid_size: float,
    random_state: int,
) -> Tuple[Sampler, Sampler, Dict[int, float]]:

    # extract only the classes and make them numpy array
    targets = np.array(train_dataset.targets)
    # train-valid split in a stratified manner (return splits iterator)
    skf_splits = sklearn_stratified_train_valid_splits(targets, valid_size, random_state)
    # select only the first split (can be expanded to K-fold cross validation)
    train_index, valid_index = next(skf_splits)

    if weighted:
        print('Using weighted sampler')
        # weights for each sample in ImageDataset (train: weighted, valid: all zeros); + class -> weight
        samples_weights, cls_to_weight = calculate_samples_weights(train_index, targets)
        # sampled indices (which must belong only to train_index) <-
        train_sampler = WeightedRandomSampler(samples_weights, len(train_index))
    else:
        # dict: idx -> weight
        cls_to_weight = get_train_class_weights(targets, train_index)
        # non-weighted random sampler
        train_sampler = SubsetRandomSampler(train_index)

    # cls_to_count_obs = {cls: 0 for cls in set(targets)}
    # for i in range(100):
    #     for train_idx in train_sampler:
    #         cls_to_count_obs[targets[train_idx]] += 1
    # print(cls_to_count_obs) # {0: 2501153, 1: 2497812, 2: 2501595, 3: 2499540}

    return train_sampler, SubsetRandomSampler(valid_index), cls_to_weight
