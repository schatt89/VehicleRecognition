import numpy as np
import cv2

import albumentations.augmentations.functional as F_albu
import torchvision.transforms.functional as F_pytorch

from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug as ia

class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size

    def __call__(self, sample):
        return F_pytorch.resize(sample, self.size)

class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return np.array(sample)


# class Resize(object):
#     def __init__(self, smallest_side_size, interpolation=cv2.INTER_LINEAR):
#         self.smallest_side_size = smallest_side_size
#         self.interpolation = interpolation
#     def __call__(self, sample):
#         print(F_albu.smallest_max_size(sample, self.smallest_side_size, self.interpolation).shape)
#         return F_albu.smallest_max_size(sample, self.smallest_side_size, self.interpolation)


# class CenterCrop(object):
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#     def __call__(self, sample):
#         return F_albu.center_crop(sample, self.height, self.width)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if np.random.rand() < self.p:
            return F_albu.hflip(sample)
        else:
            return sample


# class ToPIL(object):
#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         return Image.fromarray(sample)

def CenterCrop(height, width):
    crop = iaa.CropToFixedSize(height=height, width=width)
    crop.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    pad = iaa.PadToFixedSize(height=height, width=width, pad_mode=ia.ALL, pad_cval=(0, 255))
    pad.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    return iaa.Sequential([crop, pad])

class ImgAugTransform(object):
    
    def __init__(self, model_input, p=0.25):
        self.model_input = model_input

        self.aug = iaa.Sequential([
            iaa.Sometimes(p, 
                iaa.OneOf([
                    iaa.MotionBlur(k=15, angle=[-135, -90, -45, 45, 90, 135]),
                    iaa.GaussianBlur(sigma=(0, 3.0)),
                ])
            ),
            iaa.Sometimes(p, iaa.Affine(rotate=(-20, 20), mode='symmetric')),
            iaa.Sometimes(p, iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
            iaa.Sometimes(p, iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
            iaa.Sometimes(p, iaa.Sharpen(alpha=0.5)),
            iaa.Sometimes(p, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
            iaa.Sometimes(p, iaa.Add((-20, 20), per_channel=0.5)),
            iaa.Sometimes(p, iaa.PiecewiseAffine(scale=(0.01, 0.02))),
            iaa.Sometimes(p, iaa.PerspectiveTransform(scale=(0.01, 0.15))),
            # iaa.ChannelShuffle(p/5)
            
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)
