# Calibration-Test
This repository contains the implementation of the paper: T-Cal: An optimal test for the calibration of predictive models.
T-Cal is a minimax optimal test for calibration based on a debiased plug-in estimator of the l2-Expected Calibration Error (ECE).

## Motivation

**TLDR:** The currently prevalent metrics, such as the empirical ECE, can be misleading.
T-Cal is a test for calibration with provable guarantee.

**Long:** 

The prediction accuracy of machine learning methods is steadily increasing, but the calibration of their
uncertainty predictions poses a significant challenge. Numerous works focus on obtaining well-calibrated
predictive models, but less is known about reliably assessing model calibration. This limits our ability
to know when algorithms for improving calibration have a real effect, and when their improvements
are merely artifacts due to random noise in finite datasets. In this work, we consider detecting mis-
calibration of predictive models using a finite validation dataset as a hypothesis testing problem. The
null hypothesis is that the predictive model is calibrated, while the alternative hypothesis is that the
deviation from calibration is sufficiently large.

We find that detecting mis-calibration is only possible when the conditional probabilities of the
classes are sufficiently smooth functions of the predictions. When the conditional class probabilities
are H ̈older continuous, we propose T-Cal, a minimax optimal test for calibration based on a debiased
plug-in estimator of the l2-Expected Calibration Error (ECE). We further propose Adaptive T-Cal, a
version that is adaptive to unknown smoothness. We verify our theoretical findings with a broad range of
experiments, including with several popular deep neural net architectures and several standard post-hoc
calibration methods. T-Cal is a practical general-purpose tool, which—combined with classical tests for
discrete-valued predictors—can be used to test the calibration of virtually any probabilistic classification
method.
