# Calibration-Test (T-Cal)
<!-- This repository contains the implementation of the paper: [T-Cal: An optimal test for the calibration of predictive models.](add the link to our paper) -->
This repository contains the implementation of the paper: **T-Cal: An optimal test for the calibration of predictive models.**

T-Cal is an **adaptive** and  **minimax optimal** test for calibration of predictive models (including deep neural nets) based on a **debiased plug-in estimator** of the l2-Expected Calibration Error (ECE).


## Overview of T-Cal
For a given probability predictor, we compute a debiased plug-in
estimator (DPE)  of the ell-2 Expected Calibration Error (ECE), binned over several scales.

<img src = "https://github.com/xmhuang18/images/blob/063fa6819cdf202c4263011e994df7ea51dc765b/Calibration-Test/overview.png" width = "800">

## Motivation

**TL;DR:** The currently prevalent metrics to evaluate calibration, including the empirical ECE, can be suboptimal.
T-Cal is an optimal test for calibration with provable guarantees.


**Left:** A graph of the calibration curve, corresponding to a mis-calibrated
probability predictor. **Middle: The emprical l2-ECE values of a mis-calibrated predictor (P1) can be smaller than those of a perfectly calibrated predictor (P0).** **Right: This misleading effect is eliminated by the debiasing in T-Cal.**

<img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/altdist.png" width = "250" height = "230"> <img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/biased.png" width = "250" height = "230"> <img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/debiased.png" width = "250" height = "230">


**Long:** 

The prediction accuracy of machine learning methods is steadily increasing, but the calibration of their
uncertainty predictions poses a significant challenge. Numerous works focus on obtaining well-calibrated
predictive models, but less is known about reliably assessing model calibration. This limits our ability
to know when algorithms for improving calibration have a real effect, and when their improvements
are merely artifacts due to random noise in finite datasets. We consider detecting mis-calibration 
of predictive models using a finite validation dataset as a hypothesis testing problem. 

We find that detecting mis-calibration is only possible when the conditional probabilities of the
classes are sufficiently smooth functions of the predictions. When the conditional class probabilities
are Holder continuous, we propose **T-Cal**, a **minimax optimal** test for calibration based on a **debiased
plug-in estimator** of the l2-Expected Calibration Error (ECE). 

We further propose **Adaptive T-Cal**, a
version that is adaptive to unknown smoothness. T-Cal is a **practical general-purpose tool**, which—combined with classical tests for
discrete-valued predictors—**can be used to test the calibration of virtually any probabilistic classification
method.**

The test results of neural networks trained on Cifar-100 are below.
<img src = "https://github.com/xmhuang18/images/blob/063fa6819cdf202c4263011e994df7ea51dc765b/Calibration-Test/cifar100_table.png" width = "900">


## Usage/Reproducibility

We require the installation of the uncertainty-calibration package developed by (Kumar et al., 2019).

```sh
pip3 install uncertainty-calibration
```
All results in our paper can be reproduced with the code in `experiment/`.

Confidence scores and prediction results we used in the empirical dataset experiments (Section 4.2 of our paper) are stored in `experiments/data/`.

Reproducing our results is as simple as:
```sh
python empirical_datasets.py --model <model_name> --method <calibration_method>
```

We provide the results for
- **9 Models:** DenseNet 121 (cifar10_densenet121), ResNet 50 (cfiar10_resnet50), VGG-19 (cifar10_vgg19_bn), MobileNet-v2 (cifar100_mobilenetv2_x1_4), ResNet 56 (cifar100_resnet56), ShuffleNet-v2 (cifar100_shufflenetv2_x2_0), DenseNet 161 (imagenet_densenet161), ResNet 152 (imagenet_resnet152), Efficientnet_b7 (imagenet_efficientnet_b7)
- **6 Calibration Types:** No Calibration (none), Platt Scaling (platt), Polynomial Scaling (poly), Isotonic Regression (isotonic), Histogram Binning (histogram), Scaling Binning (scalebin)

For a full sweep, run:
```sh
bash sweep.sh
```


<!-- ## Citation

If you use T-Cal, please consider citing: -->

## Contact

Should you have any questions, please feel free to contact the authors:
Donghwan Lee, dh7401@sas.upenn.edu;
Xinmeng Huang, xinmengh@sas.upenn.edu;

