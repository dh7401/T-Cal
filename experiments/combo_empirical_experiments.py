import torch
from torch.nn import functional as F

import numpy as np
import random

import scipy.io as sio
from scipy.stats import rankdata

from calibration import PlattCalibrator, PolyCalibrator, HistogramCalibrator, PlattBinnerCalibrator
from sklearn.isotonic import IsotonicRegression



# Users are required to change the model_name and dataset by hand.
# Using a testing loop for all methods can take a long time, we thus suggest users
# to try each calibration method one by one.

# User Option: 
#                                                             
# dataset                               model_name
# cifar10:      densenet121             resnet50            vgg19_bn
# cifar100:     mobilenetv2_x1_4        resnet56            shufflenetv2_x2_0
# imagenet:     densenet161_compact     resnet152_compact   efficientnet_b7_compact
#
# calibration_method: 'No Calibration', 'Platt Scaling', 'Polynomial Scaling', 'Histogram Binning', 'Scaling Binning', 'Isotonic Regression'.

# A kind reminder is that adaptive T-Cal for models trained on ImageNet takes longer time. 
# Users can choose a smaller number of bootstrapping time, which, however, leads to less 
# stable testing results.

dataset, model_name = 'cifar10', 'densenet121'
calibration_method = 'No Calibration'
two_sided_for_miller = True




root = "empirical prediction output/"+dataset+'/'+dataset+'_'+model_name+'.mat'
mat_file = sio.loadmat(root)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2022)

def ece(scores, labels, num_bins = 15, p = 2, debias = False):
    '''
    Return biased/debiased lp-ECE_n
    This function only suits binary classificaion or scores binaried via top1 confidence,
    meanign that the label should be either 1 or 0 and the score is 1-d.
    We overwrite the ece computing function for our specific usage and clarity.
    '''
  
    # binning with equal width
    indexes = np.floor(num_bins * scores).astype(int) 
    # reindex the bins to [0,min(num_bins, sample_soze)]
    # to reduce the computation later
    indexes = rankdata(indexes, method='dense')-1
    counts = np.bincount(indexes)
    idx_real = (counts>1)

    if p == 2:
        if debias:
            error = (((np.bincount(indexes, weights=scores-labels))**2
                - np.bincount(indexes, weights=(scores-labels)**2))[idx_real] / counts[idx_real] ).sum()/ scores.shape[0]
        else:
            error = ((np.bincount(indexes, weights=scores-labels))**2[idx_real] / counts[idx_real] ).sum()/ scores.shape[0]
        
    else:
        # return plug-in lp ece without debiasing (Guo et al.)
        error = (np.abs((np.bincount(indexes, weights=scores-labels))[idx_real]/counts[idx_real])**p * counts[idx_real]).sum() / scores.shape[0]
    return error


def ece_kumar(scores, labels):
    '''
    Return debiased l2-ECE_n by Kumar et al.'s approach.
    This function only suits binary classificaion with a finte range of confidence.
    '''
    indexes = rankdata(scores, method='dense')-1
    counts = np.bincount(indexes)
    # if np.min(counts) <= 1:
        # print('Every discrete value should have at least 2 datapoints for debiased estimator!')
    acc_per_bin = np.bincount(indexes, weights=labels) / counts

    error1 = ((np.bincount(indexes, weights=scores-labels)**2) / counts / scores.shape[0]).sum()
    # smooth the denominator when a bin only contains less than 2 datapoints
    error2 = (acc_per_bin[counts>1]*(1-acc_per_bin[counts>1])/(counts[counts>1]-1)*counts[counts>1]).sum()/scores.shape[0]

    error = error1- error2
    return error


def consistency_resampling(scores):
    '''
    A bootstrapping function.
    '''
    n = scores.shape[0]
    sampled_scores = np.random.choice(scores, size=n, replace=True)
    sampled_labels = np.random.binomial(1, sampled_scores, n)
    return sampled_scores, sampled_labels


def adaptive_T_Cal(scores, labels, alpha = 0.05):
    '''
    Return True if the null hypothesis of perfect calibration is rejected, 
    and vice versa.
    The procedure is performed based on adaptive T-Cal test.
    '''
    
    reject_tag = False
    n = len(labels)
    B = np.floor(2*np.log2(n/np.sqrt(np.log(n)))).astype('int')

    for b in range(1,B+1): 
        num_bins = 2**b
        MC_dpe = np.zeros(3000,)
        for t in range(3000):
            MC_scores, MC_labels = consistency_resampling(scores)
            MC_dpe[t] = ece(MC_scores, MC_labels, num_bins = num_bins, p = 2, debias = True)
        test_dpe = ece(scores, labels, num_bins = num_bins, p = 2, debias = True)
        threshhold = np.quantile(MC_dpe, 1-alpha/B)

        reject_tag = (test_dpe > threshhold)
        if reject_tag: break
    return reject_tag


def miller_chisquare_stat(scores, labels, discrete_values):
    '''Return Miller's chi-squared statistic.'''
    miller_stat = 0.
    for value in discrete_values:
        idx = (scores == value)
        if np.sum(idx) != 0: 
            # truncate to avoid explosion
            safe_value = max(min(value, 0.99),0.01)
            miller_stat = miller_stat + np.sum((scores[idx] - labels[idx])**2)/(np.sum(idx)*safe_value*(1-safe_value))
    return miller_stat


def miller_chi_squared_test(scores, labels,  discrete_values, two_sided = True, alpha = 0.05):
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
    if two_sided:
        threshhold_u = np.quantile(MC_miller_stat, 1-alpha/2)
        threshhold_l = np.quantile(MC_miller_stat,alpha/2)
        reject_tag = (test_miller_stat > threshhold_u) or (test_miller_stat < threshhold_l)
    else:
        threshhold = np.quantile(MC_miller_stat, 1-alpha)
        reject_tag = (test_miller_stat > threshhold)
    return reject_tag





# processing the data
# the predictions have been random shuffled before being stored
# so we do not shuffle again here for reproducibility
if dataset == 'cifar10':
    logits, labels = mat_file['logit'], mat_file['label']
    poly_dg = 3
    num_cal = 2000

    # binarize the lable and score via top1 score
    bi_scores = np.zeros((10000,))
    bi_labels = np.zeros((10000,), dtype = 'int')

    softmaxes = F.softmax(torch.tensor(logits), dim = 1)
    bi_scores, bi_predictions = torch.max(softmaxes, 1)
    bi_scores = bi_scores.numpy()
    bi_labels = bi_predictions.eq(torch.tensor(labels.reshape(10000,))).numpy()*1

elif dataset == 'cifar100':
    mat_file = sio.loadmat(root)
    logits, labels = mat_file['logit'], mat_file['label']
    poly_dg = 5
    num_cal = 2000

    # binarize the lable and score via top1 score
    bi_scores = np.zeros((10000,))
    bi_labels = np.zeros((10000,), dtype = 'int')

    softmaxes = F.softmax(torch.tensor(logits), dim = 1)
    bi_scores, bi_predictions = torch.max(softmaxes, 1)
    bi_scores = bi_scores.numpy()
    bi_labels = bi_predictions.eq(torch.tensor(labels.reshape(10000,))).numpy()*1
elif dataset == 'imagenet':
    mat_file = sio.loadmat(root)
    top_1_scores, top_1_predicts, labels = mat_file['top_1_score'], mat_file['top_1_predict'], mat_file['label']
    poly_dg = 5
    num_cal = 10000

    # binarize the lable and score via top1 score
    bi_scores = top_1_scores.reshape((-1,))
    bi_labels = ((top_1_predicts==labels)*1).reshape((-1,))
else:
    raise ValueError("The dataset must be input correctly!")






if calibration_method == 'No Calibration':
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores, bi_labels, num_bins = 15, p = 1), '!')
    reject_tag = adaptive_T_Cal(bi_scores, bi_labels)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

elif calibration_method == 'Platt Scaling':
    calibrator = PlattCalibrator(num_cal, num_bins=15)
    calibrator.train_calibration(bi_scores[:num_cal], bi_labels[:num_cal])
    bi_scores_calibrated, bi_labels_calibrated = calibrator.calibrate(bi_scores[num_cal::]), bi_labels[num_cal::]
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores_calibrated, bi_labels_calibrated, num_bins = 15, p = 1), '!')
    reject_tag = adaptive_T_Cal(bi_scores_calibrated, bi_labels_calibrated)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

elif calibration_method == 'Polynomial Scaling':
    calibrator = PolyCalibrator(num_cal, num_bins=15, poly_degree = poly_dg)
    calibrator.train_calibration(bi_scores[:num_cal], bi_labels[:num_cal])
    bi_scores_calibrated, bi_labels_calibrated = calibrator.calibrate(bi_scores[num_cal::]), bi_labels[num_cal::]
    # bounding the calibrated score into [0,1]
    for i in range(len(bi_scores_calibrated)):
        if bi_scores_calibrated[i] <= 0.: bi_scores_calibrated[i] = 0.
        if bi_scores_calibrated[i] >= 1.: bi_scores_calibrated[i] = 1.
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores_calibrated, bi_labels_calibrated, num_bins = 15, p = 1), '!')
    reject_tag = adaptive_T_Cal(bi_scores_calibrated, bi_labels_calibrated)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

elif calibration_method == 'Histogram Binning':
    calibrator = HistogramCalibrator(num_cal, num_bins=15)
    calibrator.train_calibration(bi_scores[:num_cal], bi_labels[:num_cal])
    bi_scores_calibrated, bi_labels_calibrated = calibrator.calibrate(bi_scores[num_cal::]), bi_labels[num_cal::]
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores_calibrated, bi_labels_calibrated, num_bins = 15, p = 1), '!')
    print(f"The debiased empirical l2-ECE computed by Kumar et al.'s method:", ece_kumar(bi_scores_calibrated, bi_labels_calibrated), '!')
    discrete_values = np.unique(calibrator.calibrate(bi_scores[:num_cal]))
    reject_tag = miller_chi_squared_test(bi_scores_calibrated, bi_labels_calibrated,  discrete_values, two_sided = two_sided_for_miller)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

elif calibration_method == 'Scaling Binning':
    calibrator = PlattBinnerCalibrator(num_cal, num_bins=15)
    calibrator.train_calibration(bi_scores[:num_cal], bi_labels[:num_cal])
    bi_scores_calibrated, bi_labels_calibrated = calibrator.calibrate(bi_scores[num_cal::]), bi_labels[num_cal::]
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores_calibrated, bi_labels_calibrated, num_bins = 15, p = 1), '!')
    print(f"The debiased empirical l2-ECE computed by Kumar et al.'s method:", ece_kumar(bi_scores_calibrated, bi_labels_calibrated), '!')
    discrete_values = np.unique(calibrator.calibrate(bi_scores[:num_cal]))
    reject_tag = miller_chi_squared_test(bi_scores_calibrated, bi_labels_calibrated,  discrete_values, two_sided = two_sided_for_miller)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

elif calibration_method == 'Isotonic Regression':
    iso_reg = IsotonicRegression(y_min = 0., y_max = 1., increasing=True, out_of_bounds='clip')
    iso_reg.fit(bi_scores[:num_cal], bi_labels[:num_cal])
    bi_scores_calibrated, bi_labels_calibrated = iso_reg.predict(bi_scores[num_cal::]), bi_labels[num_cal::]
    print(f"The empirical l1-ECE computed by Guo et al.'s method:", ece(bi_scores_calibrated, bi_labels_calibrated, num_bins = 15, p = 1), '!')
    print(f"The debiased empirical l2-ECE computed by Kumar et al.'s method:", ece_kumar(bi_scores_calibrated, bi_labels_calibrated), '!')
    discrete_values = np.unique(iso_reg.predict(bi_scores[:num_cal]))
    reject_tag = miller_chi_squared_test(bi_scores_calibrated, bi_labels_calibrated,  discrete_values, two_sided = two_sided_for_miller)
    if reject_tag:
        print('The test result: reject!')
    else:
        print('The test result: accept!')

else: 
    raise ValueError("The calibration method must be input correctly!")
