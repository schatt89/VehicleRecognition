import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from time import localtime, strftime
import os

from train import train_model
from test import test_model
from loaders import get_train_valid_loader, get_test_loader
from models import initialize_model
from utils import workdir_copy

from transforms import ImgAugTransform

# PyTorch Version:  1.3.1
# Torchvision Version:  0.4.2
# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)


def main():
    '''
    Run as: (python ./part2/main_two_stage.py 2>&1) | tee /home/hdd/logs/openimg/$(date +'%y%m%d%H%M%S').txt
    '''
    # save the experiment time
    start_time = strftime("%y%m%d%H%M%S", localtime())

    # checks and logs
    pwd = os.getcwd()
    assert os.getcwd().endswith('VehicleRecognition')
    assert os.path.exists('./part2/experiments/')

    print(f'Working dir: {pwd}')
    # fix the random seed
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # paths to dataset
    train_data_dir = '/home/nvme/data/openimg/train/train/'
    test_data_dir = '/home/nvme/data/openimg/test/testset/'

    # Number of classes in the dataset
    num_classes = len(os.listdir(train_data_dir))

    # define the paths
    save_pred_path = None
    save_pred_path = f'./part2/experiments/{start_time}.csv'
    save_best_model_path = f'/home/hdd/logs/openimg/{start_time}/best_model.pt'

    # backup the working directiory
    workdir_copy(pwd, os.path.split(save_best_model_path)[0])

    # resnet, alexnet, vgg, squeezenet, densenet, inception
    # 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x48d_wsl', 'resnext101_32x32d_wsl'
    # 'resnext101_32x16d_wsl
    model_name = "resnext101_32x16d_wsl"

    device = torch.device("cuda:2")
    valid_size = 0.25
    if model_name.startswith('resnext'):
        lr_stage_1 = 1e-4
        lr_stage_2 = 1e-8
        batch_size_stage_1 = 32
        batch_size_stage_2 = 8
    elif model_name.startswith('densenet'):
        lr_stage_1 = 1e-5
        lr_stage_2 = 1e-8
        batch_size_stage_1 = 32
        batch_size_stage_2 = 8
    else:
        lr_stage_1 = 1e-4
        lr_stage_2 = 1e-8
        batch_size_stage_1 = 64
        batch_size_stage_2 = 16
    num_workers = 16
    pin_memory = True
    weighted_train_sampler = False
    weighted_loss = False
    # num_epochs = 40
    num_epochs_stage_1 = 10
    num_epochs_stage_2 = 15

    # preventing pytorch from allocating some memory on default GPU (0)
    torch.cuda.set_device(device)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract=True, use_pretrained=True)

    # Data augmentation and normalization for training
    # Just normalization for validation
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # albumentations
            ImgAugTransform(input_size, 0.25),
            # here they end
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }

    # Send the model to GPU
    model_ft = model_ft.to(device)

    ### STAGE-1
    train_loader, valid_loader = get_train_valid_loader(
        train_data_dir, batch_size_stage_1, data_transforms, seed, weighted_train_sampler,
        valid_size=valid_size, shuffle=True, show_sample=True, num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = get_test_loader(
        test_data_dir, batch_size_stage_1, data_transforms, num_workers=num_workers, pin_memory=pin_memory)

    dataloaders_dict = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=lr_stage_1)

    # Setup the loss fxn
    if weighted_loss:
        print('Weighted Loss')
        # {0: 0.010101, 1: 0.006622, 2: 0.0008244, 3: 0.00015335, 4: 0.0006253, 5: 0.00019665,
        # 6: 0.02631, 7: 0.00403, 8: 0.001996, 9: 0.01818, 10: 0.0004466, 11: 0.008771, 12: 0.01087,
        # 13: 0.006493, 14: 0.0017, 15: 0.000656, 16: 0.001200}
        cls_to_weight = train_loader.dataset.cls_to_weight
        weights = torch.FloatTensor([cls_to_weight[c] for c in range(num_classes)]).to(device)
    else:
        weights = torch.FloatTensor([1.0 for c in range(num_classes)]).to(device)

    criterion = nn.CrossEntropyLoss(weights)

    # print some things here so it will be seen in terminal for longer time
    print(f'Timestep: {start_time}')
    print(f'using model: {model_name}')
    print(f'Using optimizer: {optimizer_ft}')
    print(f'Device {device}')
    print(f'Batchsize: {batch_size_stage_1}, {batch_size_stage_2}')
    print(f'Transforms: {data_transforms}')

    # Train and evaluate
    print('STAGE-1')
    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, device, 
        save_best_model_path.replace('.pt', '_stage_1.pt'),
        num_epochs=num_epochs_stage_1, is_inception=(model_name == "inception")
    )

    # # do test inference
    # if save_pred_path is not None:
    #     test_model(model_ft, dataloaders_dict, device, save_pred_path,
    #                is_inception=(model_name == "inception"))


    ### STAGE-2
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    for param in model_ft.parameters():
        param.requires_grad = True
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=lr_stage_2)

    train_loader, valid_loader = get_train_valid_loader(
        train_data_dir, batch_size_stage_2, data_transforms, seed, weighted_train_sampler,
        valid_size=valid_size, shuffle=True, show_sample=True, num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = get_test_loader(
        test_data_dir, batch_size_stage_2, data_transforms, num_workers=num_workers, pin_memory=pin_memory)

    dataloaders_dict = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # Train and evaluate
    print('STAGE-2')
    print(f'Timestep: {start_time}')
    print(f'using model: {model_name}')
    print(f'Using optimizer: {optimizer_ft}')
    print(f'Device {device}')
    print(f'Batchsize: {batch_size_stage_1}, {batch_size_stage_2}')
    print(f'Transforms: {data_transforms}')
    model_ft, hist = train_model(
        model_ft, dataloaders_dict, criterion, optimizer_ft, device, 
        save_best_model_path.replace('.pt', '_stage_2.pt'),
        num_epochs=num_epochs_stage_2, is_inception=(model_name == "inception")
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
