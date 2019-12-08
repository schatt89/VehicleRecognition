import os
from glob import glob

from PIL import Image
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
