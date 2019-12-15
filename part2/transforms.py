import numpy as np
from imgaug import augmenters as iaa

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
