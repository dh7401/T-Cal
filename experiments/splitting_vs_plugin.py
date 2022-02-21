import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from effect_of_debiasing import simulate_plugin


def simulate_splitting(n, m, s, rho, num_trials, perturb_m=None):
  splitting_stats = []

  if perturb_m is None:
    for _ in range(num_trials):
      z = np.random.uniform(size=n)
      null_y = np.random.binomial(1, z, n)
      null_v, null_w = rejection_sampling(z, null_y)
      splitting_stats.append(chi_squared(null_v, null_w, m))
  
  else:
    for _ in range(num_trials):
      z = np.random.uniform(size=n)
      perturbed_z = perturb_scores(z, perturb_m, s, (-1) ** np.floor(2 * perturb_m * (z - 0.25)), rho)
      alt_y = np.random.binomial(1, perturbed_z, n)
      alt_v, alt_w = rejection_sampling(z, alt_y)
      splitting_stats.append(chi_squared(alt_v, alt_w, m))

  return splitting_stats


if __name__ == '__main__':
  np.random.seed(42) # reproducibility

  n = 20000
  s = .6
  m_star = int(n**(2 / (4*s + 1)))
  rho = 100
  ms = range(40, 250, 10) 
  eces = [0.0098 * rho * m**(-s) for m in ms] # numerically computed the L2 norm of zeta

  # plug-in test
  plugin_null_stats = simulate_plugin(n, m_star, s, rho, True, 1000)
  plugin_null_stats.sort()
  critical_val = plugin_null_stats[-50]

  plugin_t2e_means = []
  plugin_t2e_stds = []
  for m in ms:
    t2e = []
    for _ in range(3):
      plugin_alt_stats = simulate_plugin(n, m_star, s, rho, True, 1000, m)
      plugin_alt_stats.sort()
      t2e.append(np.searchsorted(plugin_alt_stats, critical_val) / 1000)

    plugin_t2e_means.append(np.mean(t2e))
    plugin_t2e_stds.append(np.std(t2e))


  # sample splitting test
  splitting_null_stats = simulate_splitting(n, m_star, s, rho, 1000)
  splitting_null_stats.sort()
  critical_val = splitting_null_stats[-50]

  splitting_t2e_means = []
  splitting_t2e_stds = []

  for m in ms:
    t2e = []
    for _ in range(3):
      splitting_alt_stats = simulate_splitting(n, m_star, s, rho, 1000, m)
      splitting_alt_stats.sort()
      t2e.append(np.searchsorted(splitting_alt_stats, critical_val) / 1000)

    splitting_t2e_means.append(np.mean(t2e))
    splitting_t2e_stds.append(np.std(t2e))


  # create plot
  pyplot_setup()

  plt.figure(figsize=(1.7, 1.7))
  plt.xlabel('$\ell_2$-ECE')
  plt.ylabel('Type II error')
  plt.errorbar(eces, plugin_t2e_means, yerr=plugin_t2e_stds, label='debiased plug-in', color='b', linewidth=0.5)
  plt.errorbar(eces, splitting_t2e_means, yerr=splitting_t2e_stds, label='sample splitting', color='r', linewidth=0.5)
  plt.axhline(y=0.95, color='k', linestyle='--', linewidth=0.5)
  plt.legend(framealpha=0.5, loc='lower left', prop={'size': 6})
  plt.savefig('figures/plugin_vs_splitting.pgf', bbox_inches = 'tight')
