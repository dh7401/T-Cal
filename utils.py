def ece(scores, labels, num_bins, p=2, debias=False):
  '''
  Return lp-ECE_n
  '''
  indexes = np.floor(num_bins * scores).astype(int)
  counts = np.bincount(indexes)
  counts[counts == 0] += 1

  if p == 2 and debias:
    error = (((np.bincount(indexes, weights=scores) - np.bincount(indexes, weights=labels))**2
              - np.bincount(indexes, weights=(scores - labels)**2)) / counts).sum()
  else:
    error = (np.abs(np.bincount(indexes, weights=scores) - np.bincount(indexes, weights=labels))**p / counts).sum()

  return error / len(scores)


def perturb_scores(scores, num_bins, smoothness, signs, scale):
  '''
  Return perturbed scores
  '''
  is_inner = (0.25 <= scores) & (scores < 0.75)
  bump = lambda x: scale * np.exp(-1 / x / (1 - x))
  rescale = lambda x: 2 * num_bins * (x - 0.25) - np.floor(2 * num_bins * (x - 0.25))
  perturbations = is_inner * num_bins**(-smoothness) * bump(rescale(scores)) * signs
  
  return scores + perturbations


def chi_squared_statistic(samples1, samples2, num_bins):
  '''
  Two-sample chi-squared statistic
  '''
  n1 = len(samples1)
  n2 = len(samples2)

  indexes1 = np.floor(num_bins * samples1).astype(int)
  indexes2 = np.floor(num_bins * samples2).astype(int)

  error = ((np.bincount(indexes1, minlength=num_bins) / n1 - np.bincount(indexes2, minlength=num_bins) / n2)**2).sum()

  return error


def rejection_sampling(scores, labels):
  '''
  Perform rejection sampling based on labels/pseudo-labels for the first/second half of given scores
  '''
  size = len(scores)
  
  scores1 = scores[:size//2]
  labels1 = labels[:size//2]
  scores2 = scores[size//2:]
  labels2 = np.random.binomial(1, scores2)
  idx1 = labels1 == 1
  idx2 = labels2 == 1
  
  return scores1[idx1], scores2[idx2]
  