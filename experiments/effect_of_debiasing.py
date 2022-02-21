import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from utils import *


def simulate_plugin(n, s, rho, debias, num_trials):
  m_star = int(n**(2 / (4*s + 1)))

  null_stats = []
  alt_stats = []
  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    null_y = np.random.binomial(1, z, n)
    null_stats.append(plugin_ece(z, null_y, m_star, debias=debias))

    perturbed_z = perturb_scores(z, m_star//2, s, 1 - 2*(z <= 0.5), rho)
    alt_y = np.random.binomial(1, perturbed_z, n)
    alt_stats.append(plugin_ece(z, alt_y, m_star, debias=debias))

  return null_stats, alt_stats


if __name__ == '__main__':
  # reproducibility
  np.random.seed(42)

  n = 10000
  s = 0.3
  rho = 100
  num_trials = 1000

  null_ece2b, alt_ece2b = simulate_plugin(n, s, rho, False, num_trials)
  null_ece2d, alt_ece2d = simulate_plugin(n, s, rho, True, num_trials)

  pyplot_setup()

  plt.figure(figsize=(1.7, 1.7))
  plt.hist(null_ece2b, label='$P_0$', alpha=0.5, bins=20, color='b')
  plt.hist(alt_ece2b, label='$P_1$', alpha=0.5, bins=20, color='r')
  plt.axvline(np.mean(null_ece2b), color='b', linestyle='dashed')
  plt.axvline(np.mean(alt_ece2b), color='r', linestyle='dashed')
  plt.xlabel('$T_{m, n}^{\mathrm{b}}$')
  plt.ylabel('count')
  plt.legend(framealpha=0.5, loc='upper right', prop={'size': 6})
  plt.savefig('figures/biased.pgf', bbox_inches = "tight")

  plt.figure(figsize=(1.7, 1.7))
  plt.hist(null_ece2d, label='$P_0$', alpha=0.5, bins=20, color='b')
  plt.hist(alt_ece2d, label='$P_1$', alpha=0.5, bins=20, color='r')
  plt.axvline(np.mean(null_ece2d), color='b', linestyle='dashed')
  plt.axvline(np.mean(alt_ece2d), color='r', linestyle='dashed')
  plt.xlabel('$T_{m, n}^{\mathrm{d}}$')
  plt.ylabel('count')
  plt.legend(framealpha=0.5, loc='upper right', prop={'size': 6})
  plt.savefig('figures/debiased.pgf', bbox_inches = "tight")
