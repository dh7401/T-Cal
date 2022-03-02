import sys 
sys.path.append('..')
import argparse
import pickle

import numpy as np
from scipy.stats import rankdata
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


def miller_chisquare_stat(scores, labels, discrete_values):
  '''Return Miller's chi-squared statistic.'''
  miller_stat = 0.
  for value in discrete_values:
    idx = (scores == value)
    if np.sum(idx): 
      # truncate to avoid explosion
      safe_value = max(min(value, 0.99), 0.01)
      miller_stat = miller_stat + np.sum((scores[idx] - labels[idx])**2)/(np.sum(idx)*safe_value*(1-safe_value))
  return miller_stat


def miller_chi_squared_test(scores, labels,  discrete_values, alpha=0.05):
  '''
  Return True if the null hypothesis of perfect calibration is rejected, 
  and vice versa.
  The procedure is performed based on Miller's test.
  '''

  reject_tag = False

  MC_miller_stat = np.zeros(10000,)
  for t in range(10000):
    MC_scores, MC_labels = consistency_resampling(scores)
    MC_miller_stat[t] = miller_chisquare_stat(MC_scores, MC_labels, discrete_values)
  test_miller_stat = miller_chisquare_stat(scores, labels,discrete_values)
  if True:
    threshhold_u = np.quantile(MC_miller_stat, 1 - alpha/2)
    threshhold_l = np.quantile(MC_miller_stat, alpha / 2)
    reject_tag = (test_miller_stat > threshhold_u) or (test_miller_stat < threshhold_l)
  else:
    threshhold = np.quantile(MC_miller_stat, 1 - alpha)
    reject_tag = (test_miller_stat > threshhold)
  return reject_tag


def ece_kumar(scores, labels):
  '''
  Return debiased l2-ECE_n by Kumar et al.'s approach.
  This function only suits binary classificaion with a finte range of confidence.
  '''
  indexes = rankdata(scores, method='dense') - 1
  counts = np.bincount(indexes)
  acc_per_bin = np.bincount(indexes, weights=labels) / counts

  error1 = ((np.bincount(indexes, weights=scores-labels)**2) / counts / scores.shape[0]).sum()
  # smooth the denominator when a bin only contains less than 2 datapoints
  error2 = (acc_per_bin[counts>1]*(1-acc_per_bin[counts>1])/(counts[counts>1]-1)*counts[counts>1]).sum()/scores.shape[0]

  return error1 - error2


if __name__ == '__main__':
  np.random.seed(2022) # reproducibility

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", 
                      choices=['cifar10_densenet121', 'cifar10_resnet50', 'cifar10_vgg19_bn',
                      'cifar100_mobilenetv2_x1_4', 'cifar100_resnet56', 'cifar100_shufflenetv2_x2_0',
                      'imagenet_densenet161_compact', 'imagenet_efficientnet_b7_compact',
                      'imagenet_resnet152_compact'], required=True)
  parser.add_argument("--method", choices=['none', 'platt', 'poly', 'histogram', 'scalebin', 'isotonic'], required=True)
  args = parser.parse_args()


  print(args.model, args.method)

  with open(f'data/{args.model}.pickle', 'rb') as f:
    scores, labels = pickle.load(f)

  poly_dg, num_cal = {'cifar10': (3, 2000), 'cifar100': (5, 2000), 'imagenet': (5, 10000)}[args.model.split('_')[0]]
  calibrator = {'none': None,
                'platt': PlattCalibrator(num_cal, num_bins=15),
                'poly': PolyCalibrator(poly_degree=poly_dg),
                'histogram': HistogramCalibrator(num_cal, num_bins=15),
                'scalebin': PlattBinnerCalibrator(num_cal, num_bins=15),
                'isotonic': IsotonicCalibrator(),
                }[args.method]
  is_discrete = args.method in ['histogram', 'scalebin', 'isotonic']

  if calibrator is None:
    print(f'Empirical l1-ECE: {plugin_ece(scores, labels, num_bins=15, p=1)}')
    test_result = adaptive_T_Cal(scores, labels)

  else:
    calibrator.train_calibration(scores[:num_cal], labels[:num_cal])
    calibrated_scores = calibrator.calibrate(scores[num_cal:])
    print(f'Empirical l1-ECE: {plugin_ece(calibrated_scores, labels[num_cal:], num_bins=15, p=1)}')
    if is_discrete:
      print(f'Debiased empirical l2-ECE: {ece_kumar(calibrated_scores, labels[num_cal:])}')
      discrete_values = np.unique(calibrator.calibrate(scores[:num_cal]))
      test_result = miller_chi_squared_test(calibrated_scores, labels[num_cal:],  discrete_values)

    else:
     test_result = adaptive_T_Cal(calibrated_scores, labels[num_cal:])

  if test_result:
    print('Test result: Reject!')
  else:
    print('Test result: Accept!')
