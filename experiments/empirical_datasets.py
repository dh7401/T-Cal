import sys 
sys.path.append('..')
import argparse
import pickle

import numpy as np
from scipy.stats import rankdata, binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from calibration import PlattCalibrator, HistogramCalibrator, PlattBinnerCalibrator

from utils import *


class PolyCalibrator:
  def __init__(self, poly_degree):
    self._poly_degree = poly_degree

  def train_calibration(self, zs, ys):
    poly = PolynomialFeatures(degree=self._poly_degree)
    poly_features = poly.fit_transform(zs.reshape((-1, 1)))
    reg = LinearRegression().fit(poly_features, ys)
    self._poly = lambda x: reg.predict(poly.fit_transform(x.reshape((-1, 1))))

  def calibrate(self, zs):
    return np.clip(self._poly(zs), 0., 1.)
  

class IsotonicCalibrator:
  def __init__(self):
    pass

  def train_calibration(self, zs, ys):
    reg = IsotonicRegression(y_min = 0., y_max = 1., out_of_bounds='clip').fit(zs, ys)
    self._reg = lambda x: reg.predict(x)
  
  def calibrate(self, zs):
    return self._reg(zs)


def multiple_binom_test(scores, labels, alpha=0.05):
  '''
  Given observations X_i ~ Bin(n_i, p_i), test p_i = q_i for all i.
  Return True if the null hypothesis is rejected.
  '''
  values = np.unique(scores)
  for value in values:
    indexes = scores == value
    p_value = binomtest(labels[indexes].sum(), indexes.sum(), value).pvalue
    if p_value <= alpha / len(values):
      return True
  
  return False


def discrete_debiased_l2_ece(scores, labels):
  '''
  Return debiased l2-ECE_n by Kumar et al.'s approach.
  This function only suits discrete predictions.
  '''
  indexes = rankdata(scores, method='dense') - 1
  counts = np.bincount(indexes)
  acc_per_bin = np.bincount(indexes, weights=labels) / counts

  error1 = ((np.bincount(indexes, weights=scores - labels)**2) / counts / scores.shape[0]).sum()
  # smooth the denominator when a bin only contains less than 2 datapoints
  error2 = (acc_per_bin[counts>1]*(1-acc_per_bin[counts>1])/(counts[counts>1]-1)*counts[counts>1]).sum()/scores.shape[0]

  return error1 - error2


if __name__ == '__main__':
  np.random.seed(2022) # reproducibility

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", 
                      choices=['cifar10_densenet121', 'cifar10_resnet50', 'cifar10_vgg19_bn',
                      'cifar100_mobilenetv2_x1_4', 'cifar100_resnet56', 'cifar100_shufflenetv2_x2_0',
                      'imagenet_densenet161', 'imagenet_resnet152', 'imagenet_efficientnet_b7'], required=True)
  parser.add_argument("--method", choices=['none', 'platt', 'poly', 'histogram', 'scalebin', 'isotonic'], required=True)
  args = parser.parse_args()

  print(
f"""
----------------------------------------------------------------------
Model: {args.model} | Calibration method: {args.method}
----------------------------------------------------------------------
""")

  with open(f'data/{args.model}.pickle', 'rb') as f:
    scores, labels = pickle.load(f)

  poly_dg, num_cal = {'cifar10': (3, 2000), 'cifar100': (5, 2000), 'imagenet': (5, 10000)}[args.model.split('_')[0]]
  calibrator = {'none': None,
                'platt': PlattCalibrator(num_cal, num_bins=15),
                'poly': PolyCalibrator(poly_degree=poly_dg),
                'isotonic': IsotonicCalibrator(),
                'histogram': HistogramCalibrator(num_cal, num_bins=15),
                'scalebin': PlattBinnerCalibrator(num_cal, num_bins=15),
                }[args.method]
  is_discrete = args.method in ['histogram', 'scalebin']

  if calibrator is None:
    print(f'Empirical l1-ECE: {100 * plugin_ece(scores, labels, num_bins=15, p=1):.2f}%')
    test_result = adaptive_T_Cal(scores, labels)

  else:
    calibrator.train_calibration(scores[:num_cal], labels[:num_cal])
    calibrated_scores = calibrator.calibrate(scores[num_cal:])
    print(f'Empirical l1-ECE: {100 * plugin_ece(calibrated_scores, labels[num_cal:], num_bins=15, p=1):.2f}%')
    if is_discrete:
      print(f'Debiased empirical l2-ECE: {100 * discrete_debiased_l2_ece(calibrated_scores, labels[num_cal:]):.2f}%')
      test_result = multiple_binom_test(calibrated_scores, labels[num_cal:])

    else:
     test_result = adaptive_T_Cal(calibrated_scores, labels[num_cal:])

  if test_result:
    print('Test result: Reject!')
  else:
    print('Test result: Accept!')
