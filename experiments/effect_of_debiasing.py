import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from utils import *


def simulate_null_plugin(n, m, debias, num_trials):
  plugin_stats = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    null_y = np.random.binomial(1, z, n)
    plugin_stats.append(plugin_ece(z, null_y, m, debias=debias))

  return plugin_stats


def simulate_alt_plugin(n, m, s, rho, debias, perturb_m, num_trials):
  plugin_stats = []
  
  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    perturbed_z = perturb_scores(z, perturb_m, s, 1 - 2*(z <= 0.5), rho)
    alt_y = np.random.binomial(1, perturbed_z, n)
    plugin_stats.append(plugin_ece(z, alt_y, m, debias=debias))

  return plugin_stats


if __name__ == '__main__':
  np.random.seed(42) # reproducibility

  n = 10000
  s = 0.3
  m_star = int(n**(2 / (4*s + 1)))
  rho = 100
  num_trials = 1000

  null_biased = simulate_null_plugin(n, m_star, False, num_trials)
  alt_biased = simulate_alt_plugin(n, m_star, s, rho, False, m_star//2, num_trials)
  null_debiased = simulate_null_plugin(n, m_star, True, num_trials)
  alt_debiased = simulate_alt_plugin(n, m_star, s, rho, True, m_star//2, num_trials)


  # create plot
  pyplot_setup()

  plt.figure(figsize=(1.7, 1.7))
  plt.hist(null_biased, label='$P_0$', alpha=0.5, bins=20, color='b')
  plt.hist(alt_biased, label='$P_1$', alpha=0.5, bins=20, color='r')
  plt.axvline(np.mean(null_biased), color='b', linestyle='dashed')
  plt.axvline(np.mean(alt_biased), color='r', linestyle='dashed')
  plt.xlabel('$T_{m^*, n}^{\mathrm{b}}$')
  plt.ylabel('Count')
  plt.legend(framealpha=0.5, loc='upper right', prop={'size': 6})
  plt.savefig('figures/biased.pgf', bbox_inches = 'tight')

  plt.figure(figsize=(1.7, 1.7))
  plt.hist(null_debiased, label='$P_0$', alpha=0.5, bins=20, color='b')
  plt.hist(alt_debiased, label='$P_1$', alpha=0.5, bins=20, color='r')
  plt.axvline(np.mean(null_debiased), color='b', linestyle='dashed')
  plt.axvline(np.mean(alt_debiased), color='r', linestyle='dashed')
  plt.xlabel('$T_{m^*, n}^{\mathrm{d}}$')
  plt.ylabel('Count')
  plt.legend(framealpha=0.5, loc='upper right', prop={'size': 6})
  plt.savefig('figures/debiased.pgf', bbox_inches = 'tight')
