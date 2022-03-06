import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm

from utils import *
from tests_comparison import simulate_null_plugin, simulate_alt_plugin


if __name__ == '__main__':
  np.random.seed(42) # reproducibility

  ns = [2000, 5000, 10000]
  s = .6
  rho = 100
  ms = range(40, 200, 10)
  eces = [0.0098 * rho * m**(-s) for m in ms] # numerically computed the L2 norm of zeta
  

  # debiased l2 test
  l2d_t2e_means_list = []
  l2d_t2e_stds_list = []

  for n in ns:
    m_star = int(n**(2 / (4*s + 1)))
    l2d_null_stats = simulate_null_plugin(n, m_star, 2, True, 1000)
    l2d_null_stats.sort()
    critical_val = l2d_null_stats[-50]

    l2d_t2e_means = []
    l2d_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        l2d_alt_stats = simulate_alt_plugin(n, m_star, s, rho, 2, True, m, 1000)
        l2d_alt_stats.sort()
        t2e.append(np.searchsorted(l2d_alt_stats, critical_val) / 1000)

      l2d_t2e_means.append(np.mean(t2e))
      l2d_t2e_stds.append(np.std(t2e))
    
    l2d_t2e_means_list.append(l2d_t2e_means)
    l2d_t2e_stds_list.append(l2d_t2e_stds)


  # biased l2 test
  l2b_t2e_means_list = []
  l2b_t2e_stds_list = []

  for n in ns:
    m_star = int(n**(2 / (4*s + 1)))
    l2b_null_stats = simulate_null_plugin(n, m_star, 2, False, 1000)
    l2b_null_stats.sort()
    critical_val = l2b_null_stats[-50]

    l2b_t2e_means = []
    l2b_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        l2b_alt_stats = simulate_alt_plugin(n, m_star, s, rho, 2, False, m, 1000)
        l2b_alt_stats.sort()
        t2e.append(np.searchsorted(l2b_alt_stats, critical_val) / 1000)

      l2b_t2e_means.append(np.mean(t2e))
      l2b_t2e_stds.append(np.std(t2e))
    
    l2b_t2e_means_list.append(l2b_t2e_means)
    l2b_t2e_stds_list.append(l2b_t2e_stds)


  # biased l1 test
  l1b_t2e_means_list = []
  l1b_t2e_stds_list = []

  for n in ns:
    m_star = int(n**(2 / (4*s + 1)))
    l1b_null_stats = simulate_null_plugin(n, m_star, 1, False, 1000)
    l1b_null_stats.sort()
    critical_val = l1b_null_stats[-50]

    l1b_t2e_means = []
    l1b_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        l1b_alt_stats = simulate_alt_plugin(n, m_star, s, rho, 1, False, m, 1000)
        l1b_alt_stats.sort()
        t2e.append(np.searchsorted(l1b_alt_stats, critical_val) / 1000)

      l1b_t2e_means.append(np.mean(t2e))
      l1b_t2e_stds.append(np.std(t2e))
    
    l1b_t2e_means_list.append(l1b_t2e_means)
    l1b_t2e_stds_list.append(l1b_t2e_stds)


  # create plot
  pyplot_setup()

  colors = ['r', 'g', 'b']
  plt.figure(figsize=(3.4, 1.7))

  for i in range(3):
    l2d_t2e_means = l2d_t2e_means_list[i]
    l2d_t2e_stds = l2d_t2e_stds_list[i]
    plt.errorbar(eces, l2d_t2e_means, yerr=l2d_t2e_stds, label=f'$n = {ns[i]}$', color = f'{colors[i]}', linewidth=0.5)

  for i in range(3):
    l2b_t2e_means = l2b_t2e_means_list[i]
    l2b_t2e_stds = l2b_t2e_stds_list[i]
    plt.errorbar(eces, l2b_t2e_means, yerr=l2b_t2e_stds, color = f'{colors[i]}', linestyle='dotted', linewidth=0.5)
  
  for i in range(3):
    l1b_t2e_means = l1b_t2e_means_list[i]
    l1b_t2e_stds = l1b_t2e_stds_list[i]
    plt.errorbar(eces, l1b_t2e_means, yerr=l1b_t2e_stds, color = f'{colors[i]}', linestyle='dashdot', linewidth=0.5)

  plt.xlabel('$\ell_2$-ECE')
  plt.ylabel('Type II error')
  line1 = Line2D([0], [0], label='Debiased $\ell_2$', color='k', linewidth=0.5)
  line2 = Line2D([0], [0], label='Biased $\ell_2$', color='k', linestyle='dotted', linewidth=0.5)
  line3 = Line2D([0], [0], label='Biased $\ell_1$', color='k', linestyle='dashdot', linewidth=0.5)
  handles, _ = plt.gca().get_legend_handles_labels()
  handles.extend([line1, line2, line3])
  plt.legend(handles=handles, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('figures/l1_vs_l2.pgf', bbox_inches = 'tight')
