#!/bin/bash

for model in cifar10_densenet121 cifar10_resnet50 cifar10_vgg19_bn cifar100_mobilenetv2_x1_4 cifar100_resnet56 cifar100_shufflenetv2_x2_0 imagenet_densenet161_compact imagenet_efficientnet_b7_compact imagenet_resnet152_compact
do
    for method in none platt poly histogram scalebin isotonic
    do
        python combo_empirical_experiments.py --model $model --method $method
    done
done