import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from splitting_vs_plugin import simulate_null_plugin, simulate_alt_plugin


def min_detection_m(n, s, rho):
  num_bins = int(n**(2 / (4*s + 1)))

  null_stats = simulate_null_plugin(n, num_bins, 100)
  null_stats.sort()
  
  # binary search
  m_lower, m_upper = 1, num_bins
  while m_lower < m_upper:
    m_guess = (m_lower + m_upper) // 2
    alt_stats = simulate_alt_plugin(n, num_bins, s, rho, m_guess, 100)
    alt_stats.sort()
    if null_stats[-5] <= alt_stats[5]:
      m_lower = m_guess + 1
    else:
      m_upper = m_guess
    
  return m_lower


if __name__ == '__main__':
  np.random.seed(42) # reproducibility

  for i, (s, rho) in enumerate([(0.3, 25), (0.5, 50)]):
    ns = [1000, 2000, 5000, 10000, 20000, 50000]
    log_ece_means = []
    log_ece_stds = []
    for n in ns:
      log_ece = np.log10([0.0098 * rho * min_detection_m(n, s, rho)**(-s) for _ in range(10)])
      log_ece_means.append(np.mean(log_ece))
      log_ece_stds.append(np.std(log_ece))

    plt.figure(figsize=(1.7, 1.7))

    plt.scatter(np.log10(ns), log_ece_means, label='Empirical', s=3, color='k')
    plt.errorbar(np.log10(ns), log_ece_means, yerr=log_ece_stds, linestyle='None', color='k', linewidth=0.5)
    plt.xlabel('$\log_{10} n$')
    plt.ylabel(r'$\log_{10} \varepsilon_n$')
    plt.plot(np.log10(ns), -2 * s * np.log10(ns) / (4*s + 1), label='Theoretical', color='r', linewidth=0.5)
    plt.legend(framealpha=0.5, loc='lower left')
    plt.savefig(f'detection_rate{i}.pgf', bbox_inches = 'tight')