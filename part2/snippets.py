from models import initialize_model
from loaders import get_train_valid_loader, get_test_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets

from typing import List, Tuple

# def calculate_class_weights(data_dir: str) -> Dict[str, float]:
#     classes = os.listdir(data_dir)
#     train_size = len(glob(os.path.join(data_dir, '*/*.jpg')))
#     class_to_weight = {}
#     for cls in classes:
#         cls_size = len(glob(os.path.join(data_dir, f'{cls}/*.jpg')))
#         class_to_weight[cls] = cls_size

#     return class_to_weight


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def define_dataset(traindir):
    dataset_train = datasets.ImageFolder(traindir)
    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(
        dataset_train.imgs, 
        len(dataset_train.classes)
    )
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True,
        sampler=sampler, num_workers=0, pin_memory=True
    )

def test_weights_in_sampler():       
    import torch
    from torch.utils.data import WeightedRandomSampler

    cls0 = [0] * 10
    cls1 = [1] * 20
    cls2 = [2] * 30
    cls3 = [3] * 40

    samples = cls0 + cls1 + cls2 + cls3
    cls_to_count = {cls: samples.count(cls) for cls in set(samples)}
    cls_weights = {cls: 1 / cls_to_count[cls] for cls, count in cls_to_count.items()}
    img_weights = [cls_weights[cls] for cls in samples]
    
    print(cls_to_count)
    print(img_weights)
    sampler = WeightedRandomSampler(img_weights, len(samples))

    cls_to_count_obs = {cls: 0 for cls in set(samples)}

    for i in range(100000):
        for sample in sampler:
            cls_to_count[samples[sample]] += 1

    return cls_to_count # {0: 2501153, 1: 2497812, 2: 2501595, 3: 2499540}

# TODO: inputs train_dataset
def make_weighted_train_valid_samplers():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import WeightedRandomSampler

    def sklearn_stratified_train_valid_split(y: np.ndarray) -> Tuple[list, list]:
        # X which is not used in .split anyway
        X = np.arange(len(y) * 2).reshape(-1, 2)
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=13)

        for train_index, test_index in skf.split(X, y):
            print("TRAIN (idx):", train_index)
            print("TRAIN (cls):", y[train_index])
            print("TEST (idx):", test_index)
            print("TEST (cls):", y[test_index])
            print()

        # select only the first split
        print(type(skf.split(X, y)))
        return next(skf.split(X, y))

    def calculate_samples_weights(
            train_index: list, 
            path_cls: List[Tuple[str, int]]
        ) -> torch.DoubleTensor:

        train_path_cls = [path_cls[idx] for idx in train_index]
        train_sample_cls = [Class for Path, Class in train_path_cls]
        cls_to_count = {cls: train_sample_cls.count(cls) for cls in set(train_sample_cls)}
        cls_to_weight = {cls: 1 / cls_to_count[cls] for cls, count in cls_to_count.items()}
        print(cls_to_weight)

        samples_weights = torch.zeros(len(path_cls)).double()

        for idx in train_index:
            sample_path, sample_class = path_cls[idx]
            samples_weights[idx] = cls_to_weight[sample_class]

        return samples_weights



    # mimics train_dataset.imgs
    import random, string
    def randomString():
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for i in range(3))
    path_cls = [(randomString(), 0) for _ in range(20)] + \
               [(randomString(), 1) for _ in range(80)] + \
               [(randomString(), 2) for _ in range(8)]
    print(path_cls)
    
    samples_classes = np.array([Class for Path, Class in path_cls])
    train_index, valid_index = sklearn_stratified_train_valid_split(samples_classes)
    samples_weights = calculate_samples_weights(train_index, path_cls)
    print(samples_weights)
    # sampled indices (which must belong only to train_index) <-
    train_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # cls_to_count_obs = {cls: 0 for cls in set(samples_classes)}
    # for i in range(100):
    #     for sample in train_sampler:
    #         cls_to_count_obs[samples_classes[sample]] += 1

    # print(cls_to_count_obs) # {0: 2501153, 1: 2497812, 2: 2501595, 3: 2499540}

    return train_sampler, valid_index


def test_samplers():
    a = []
    for i in range(1, 12):
        with open(f'./part2/sums_{i}_train.txt', 'r') as outf:
            for line in outf.readlines():
                a.append(float(line.replace('\n', '')))

    print(len(np.unique(np.array(a))))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model_name, model_path, show_only_fails, device, weighted_train_sampler, 
    num_images=10*5):
    seed = np.random.randint(100)
    train_data_dir = '/home/nvme/data/openimg/train/train/'
    test_data_dir = '/home/nvme/data/openimg/test/testset/'
    # hyper parameters
    valid_size = 0.25
    batch_size = 32
    num_workers = 4
    pin_memory = True
    num_classes = 17

    # Initialize the model for this run
    model, input_size = initialize_model(
        model_name, num_classes, feature_extract=True, use_pretrained=True)

    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    was_training = model.training
    model.eval()
    # Data augmentation and normalization for training
    # Just normalization for validation
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }

    train_loader, valid_loader = get_train_valid_loader(
        train_data_dir, batch_size, data_transforms, seed, weighted_train_sampler,
        valid_size=valid_size, shuffle=True, show_sample=False, num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = get_test_loader(
        test_data_dir, batch_size, data_transforms, num_workers=num_workers, pin_memory=pin_memory)

    dataloaders_dict = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {Class: idx for idx, Class in class_to_idx.items()}

    while True:
        images_so_far = 0
        fig = plt.figure(figsize=(20, 10))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders_dict['valid']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    predicted = idx_to_class[preds[j].item()]
                    gt = idx_to_class[labels[j].item()]
                    if show_only_fails and predicted == gt:
                        continue
                    if predicted == gt:
                        title_color = 'green'
                    else:
                        title_color = 'red'

                    images_so_far += 1
                    ax = plt.subplot(num_images//10, 10, images_so_far)
                    ax.axis('off')

                    # print(predicted, gt)
                    ax.set_title(f'p: {predicted}; gt: {gt}', fontsize=7, color=title_color)
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        break
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    break

            model.train(mode=was_training)

        plt.show()
        print('Use Ctrl-C in the Terminal to stop the infinite loop')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Training experiment.')
    # parser.add_argument('--epochs', type=int, default=10)
    # args = parser.parse_args()
    # print(args)
    # data_dir = '/home/nvme/data/openimg/train/train'
    # print(calculate_class_weights(data_dir))
    # define_dataset(data_dir)
    # print(test_weights_in_sampler())
    # print(make_weighted_train_valid_samplers())
    # test_samplers()

    visualize_model(
        model_name="resnext101_32x32d_wsl",
        model_path='/home/hdd/logs/openimg/191207181758/best_model.pt',
        show_only_fails=True,
        device=torch.device('cuda:1'),
        weighted_train_sampler=True,
        num_images=10*5
    )
