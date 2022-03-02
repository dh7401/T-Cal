# Calibration-Test
This repository contains the implementation of the paper: T-Cal: An optimal test for the calibration of predictive models.
T-Cal is a minimax optimal test for calibration based on a debiased plug-in estimator of the l2-Expected Calibration Error (ECE).


An overview of T-Cal is as below. For a given probability predictor, we compute the debiased plug-in
estimator (DPE), binned over several scales.
<img src = "https://github.com/xmhuang18/images/blob/063fa6819cdf202c4263011e994df7ea51dc765b/Calibration-Test/overview.png" width = "800">

## Motivation

**TLDR:** The currently prevalent metrics, such as the empirical ECE, can be misleading.
T-Cal is a test for calibration with provable guarantee.


**Left:** A graph of the calibration curve, correspondign to a mis-calibrated
probability predictor. **Middle: The emprical l2-ECE values of a mis-calibrated predictor (P1) can be smaller than those of a perfectly calirbated predictor (P0).** **Right: This misleading effect is elimiated by debiasing in T-Cal.**

<img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/altdist.png" width = "300" height = "280"> <img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/biased.png" width = "300" height = "280"> <img src="https://github.com/xmhuang18/images/blob/3f3b3dc11036a8a17603c1ada57c1a8c184dc0c4/Calibration-Test/debiased.png" width = "300" height = "280">


**Long:** 

The prediction accuracy of machine learning methods is steadily increasing, but the calibration of their
uncertainty predictions poses a significant challenge. Numerous works focus on obtaining well-calibrated
predictive models, but less is known about reliably assessing model calibration. This limits our ability
to know when algorithms for improving calibration have a real effect, and when their improvements
are merely artifacts due to random noise in finite datasets. We consider detecting mis-calibration of predictive models using a finite validation dataset as a hypothesis testing problem. 

We find that detecting mis-calibration is only possible when the conditional probabilities of the
classes are sufficiently smooth functions of the predictions. When the conditional class probabilities
are Holder continuous, we propose **T-Cal**, a **minimax optimal** test for calibration based on a **debiased
plug-in estimator** of the l2-Expected Calibration Error (ECE). We further propose **Adaptive T-Cal**, a
version that is adaptive to unknown smoothness. T-Cal is a **practical general-purpose tool**, which—combined with classical tests for
discrete-valued predictors—**can be used to test the calibration of virtually any probabilistic classification
method.**

Testing results of neural networks trained on Cifar-100 is as below table.
<img src = "https://github.com/xmhuang18/images/blob/063fa6819cdf202c4263011e994df7ea51dc765b/Calibration-Test/cifar100_table.png" width = "900">


## Usage/Reproducibility

First train a DenseNet on CIFAR100, and save the validation indices:
```sh
python empirical_datasets.py --model <model_name> --method <calibration_method>
```

## Contact

Should you have any questions, please feel free to contact the authors.
