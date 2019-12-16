# 5th place solution for TAU Vehicle Type Recognition Competition

This repo contains the 5th place solution for the [TAU Vehicle Type Recognition Competition](https://www.kaggle.com/c/vehicle/overview) on Kaggle which was organized by [Tampere University](https://www.tuni.fi/en) in 2019. The competition was a mandatory part of [Pattern Recognition and Machine Learning](http://www.cs.tut.fi/courses/SGN-41006/) course. 

## The team 

- Anna Iashina ([LinkedIn](https://www.linkedin.com/in/anna-iashina/))
- Einary Vaaras ([e-mail]<einari.vaaras@tuni.fi>)
- Vladimir Iashin ([LinkedIn](https://www.linkedin.com/in/vladimir-iashin/))
- Maral Zadehdarrehshoorian ([LinkedIn](https://www.linkedin.com/in/mzdarrehshoorian/))

# Solution

For the two submissions for the private leaderboard, we decided to use the best model on the local validation and the best on the public leaderboard.

The best model on local validation was a finetuned [ResNeXt 101 WSL (32x16d)](https://arxiv.org/abs/1805.00932) pre-trained on weak labels from Instagram images. We trained it on 90 % of the dataset with mild augmentation. This model solely gave us 92.75 on local validation (10 % of the dataset) and 90.8 on public lb (92.1 private (5th place)). We decided not to use this one for the final prediction as this 90/10 validation scheme wasn't tested properly and we didn't want to have an overfitted model in our final set of submissions for private lb since 10 % was still a quite small portion of data even though the split was stratified. So, instead, we used the same model but trained on 75/25 split (local: 92.3, public: 91.0, private: 91.45). Anyway, it wouldn't change our position on private lb. 

Another submission which was meant to maximize the public lb score was a majority vote of many models from previous submissions which, we believe, were orthogonal enough to each other. Specifically, it was a blend of three: another result of ResNeXt 101 WSL (32x16d) (local: 92.65); DenseNet (local: 91.45); and fused prediction of other 5 models (sklearn classifiers on ResNet features, Inception v3, Resnext 101 WSL, Efficientnet, Resnet 101).

Validation: stratified split 75/25 (we didn't do K-Fold CV to save time but you can add one loop to the code easily).

What didn't work:
1. Weighted sampler and loss
2. Two-stage training: when the model weights are frozen and only the last layer is trained and, after, the whole model is finetuned with a lower lr
3. AdamW optimizer (the same results)

# Contents description

The brief description of what has been done so far

## Part 1

The folder contains the initial Tensorflow (Keras) solutions including the ResNet101 + sklearn models which we used for the final blend and a bunch of pre-trained models with ImageNet weights. All the solutions reach up to 0.88 accuracy on validation (0.86 on public and private LB).

## Part 2

Here we provide a code (PyTorch) for finetuning a ResNeXt 101 WSL model which we found to achieve the best performance among all tested models. The best model was ResNeXt 101 WSL (32x16d) which gave 0.9275 on local validation (0.908 on public and 0.921 on private LB).

## Part 3

This part contains the code for blending the predictions from several models. Our best model on private LB (final submission) was a majority vote of the second-best model on local validation (ResNeXt 101 WSL), DenseNet, and the best model on a public LB at that time.