#!/bin/bash

for model in cifar10_densenet121 cifar10_resnet50 cifar10_vgg19_bn cifar100_mobilenetv2_x1_4 cifar100_resnet56 cifar100_shufflenetv2_x2_0 imagenet_densenet161 imagenet_resnet152 imagenet_efficientnet_b7
do
    for method in none platt poly isotonic histogram scalebin 
    do
        python empirical_datasets.py --model $model --method $method
    done
done