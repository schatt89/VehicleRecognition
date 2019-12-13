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

# PyTorch Version:  1.3.1
# Torchvision Version:  0.4.2
# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)

def main():
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
    model_name = "squeezenet"
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    # hyper parameters
    device = torch.device("cuda:2")
    valid_size = 0.25
    if model_name.startswith('resnext'):
        lr = 1e-4
        batch_size = 32
    else:
        lr = 1e-4
        batch_size = 64
    num_workers = 16
    pin_memory = True
    weighted_train_sampler = False
    weighted_loss = False
    num_epochs = 30

    # preventing pytorch from allocating some memory on default GPU (0)
    torch.cuda.set_device(device)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True)

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
        valid_size=valid_size, shuffle=True, show_sample=True, num_workers=num_workers, 
        pin_memory=pin_memory
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
    optimizer_ft = optim.Adam(params_to_update, lr=lr)

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
    print(f'Batchsize: {batch_size}')
    print(f'Transforms: {data_transforms}')


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
