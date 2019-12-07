import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from time import localtime, strftime
import os
import copy

from train import train_model
from test import test_model
from dataset import get_train_valid_loader, get_test_loader
from models import initialize_model
from utils import workdir_copy

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def main():
    # fix the random seed
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # checks
    pwd = os.getcwd()
    assert os.getcwd().endswith('VehicleRecognition')
    assert os.path.exists('./part2/experiments/')
    print(f'Working dir: {pwd}')

    # save the experiment time
    start_time = strftime("%y%m%d%H%M%S", localtime())
    print(f'Timestep: {start_time}')

    # define the paths
    save_pred_path = None
    save_pred_path = f'./part2/experiments/{start_time}.csv'
    save_best_model_path = f'/home/hdd/logs/openimg/{start_time}/best_model.pt'

    # backup the working directiory
    workdir_copy(pwd, os.path.split(save_best_model_path)[0])
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # valid ratio
    valid_size = 0.25
    # learning rate
    lr = 1e-4

    # paths to dataset
    # data_dir = '/home/nvme/data/openimg/hymenoptera_data/'
    train_data_dir = '/home/nvme/data/openimg/train/train/'
    test_data_dir = '/home/nvme/data/openimg/test/testset/'

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    # 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x48d_wsl'
    # 'resnext101_32x32d_wsl'
    model_name = "resnext101_32x32d_wsl"

    # Number of classes in the dataset
    num_classes = len(os.listdir(train_data_dir))

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    # num of workers for data loading
    num_workers = 16
    pin_memory = True
    weighted = True

    # Number of epochs to train for
    num_epochs = 20

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)
    print(f'using model: {model_name}')

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_loader, valid_loader = get_train_valid_loader(
        train_data_dir, batch_size, data_transforms, seed, weighted, valid_size=valid_size,
        shuffle=True, show_sample=True, num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = get_test_loader(
        test_data_dir, batch_size, data_transforms, num_workers=num_workers, pin_memory=pin_memory)

    dataloaders_dict = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    print('Using adam')
    optimizer_ft = optim.Adam(params_to_update, lr=lr)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, device, save_best_model_path,
        num_epochs=num_epochs, is_inception=(model_name == "inception")
    )

    # do test inference
    if save_pred_path is not None:
        test_model(model_ft, dataloaders_dict, device, save_pred_path,
                is_inception=(model_name == "inception"))



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Training experiment.')
    # parser.add_argument('--epochs', type=int, default=10)
    # args = parser.parse_args()
    # print(args)
    main()
