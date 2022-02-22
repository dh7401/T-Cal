import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression

from utils import *
from splitting_vs_plugin import simulate_null_plugin, simulate_alt_plugin


def simulate_null_logistic(n, num_trials):
  logistic_slopes = []
  logistic_intercepts = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    logit = np.log(z / (1 - z)).reshape(n, 1)
    null_y = np.random.binomial(1, z, n)
    clf = LogisticRegression().fit(logit, null_y)
    logistic_slopes.append(clf.coef_)
    logistic_intercepts.append(clf.intercept_)

  return logistic_slopes, logistic_intercepts


def simulate_alt_logistic(n, s, rho, perturb_m, num_trials):
  logistic_slopes = []
  logistic_intercepts = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    logit = np.log(z / (1 - z)).reshape(n, 1)
    perturbed_z = perturb_scores(z, perturb_m, s, (-1) ** np.floor(2 * perturb_m * (z - 0.25)), rho)
    alt_y = np.random.binomial(1, perturbed_z, n)
    clf = LogisticRegression().fit(logit, alt_y)
    logistic_slopes.append(clf.coef_)
    logistic_intercepts.append(clf.intercept_)

  return logistic_slopes, logistic_intercepts


if __name__ == '__main__':
  np.random.seed(42) # reproducibility

  ns = [2000, 5000, 10000]
  s = .6
  rho = 100
  ms = range(40, 200, 10)
  eces = [0.0098 * rho * m**(-s) for m in ms] # numerically computed the L2 norm of zeta
  

  # plug-in test with optimal binning
  plugin_t2e_means_list = []
  plugin_t2e_stds_list = []

  for n in ns:
    m_star = int(n**(2 / (4*s + 1)))
    plugin_null_stats = simulate_null_plugin(n, m_star, 1000)
    plugin_null_stats.sort()
    critical_val = plugin_null_stats[-50]

    plugin_t2e_means = []
    plugin_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        plugin_alt_stats = simulate_alt_plugin(n, m_star, s, rho, m, 1000)
        plugin_alt_stats.sort()
        t2e.append(np.searchsorted(plugin_alt_stats, critical_val) / 1000)

      plugin_t2e_means.append(np.mean(t2e))
      plugin_t2e_stds.append(np.std(t2e))
    
    plugin_t2e_means_list.append(plugin_t2e_means)
    plugin_t2e_stds_list.append(plugin_t2e_stds)


  # plug-in test with a fixed binning
  fixed_t2e_means_list = []
  fixed_t2e_stds_list = []

  for n in ns:
    fixed_null_stats = simulate_null_plugin(n, 15, 1000)
    fixed_null_stats.sort()
    critical_val = fixed_null_stats[-50]

    fixed_t2e_means = []
    fixed_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        fixed_alt_stats = simulate_alt_plugin(n, 15, s, rho, m, 1000)
        fixed_alt_stats.sort()
        t2e.append(np.searchsorted(fixed_alt_stats, critical_val) / 1000)

      fixed_t2e_means.append(np.mean(t2e))
      fixed_t2e_stds.append(np.std(t2e))
    
    fixed_t2e_means_list.append(fixed_t2e_means)
    fixed_t2e_stds_list.append(fixed_t2e_stds)


  # logistic test
  logistic_t2e_means_list = []
  logistic_t2e_stds_list = []

  for n in ns:
    null_slopes, null_intercepts = simulate_null_logistic(n, 1000)
    null_slopes.sort()
    null_intercepts.sort()

    slope_l = null_slopes[12]
    slope_r = null_slopes[-13]
    intercept_l = null_intercepts[12]
    intercept_r = null_intercepts[-13]

    logistic_t2e_means = []
    logistic_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        alt_slopes, alt_intercepts = simulate_alt_logistic(n, s, rho, m, 1000)

        count = 0
        for coef, intercept in zip(alt_slopes, alt_intercepts):
          if slope_l <= coef <= slope_r and intercept_l <= intercept <= intercept_r:
            count += 1

        t2e.append(count / 1000)
      logistic_t2e_means.append(np.mean(t2e))
      logistic_t2e_stds.append(np.std(t2e))
    
    logistic_t2e_means_list.append(logistic_t2e_means)
    logistic_t2e_stds_list.append(logistic_t2e_stds)


  # create plot
  pyplot_setup()

  colors = ['r', 'g', 'b']
  plt.figure(figsize=(3.4, 1.7))

  for i in range(3):
    plugin_t2e_means = plugin_t2e_means_list[i]
    plugin_t2e_stds = plugin_t2e_stds_list[i]
    plt.errorbar(eces, plugin_t2e_means, yerr=plugin_t2e_stds, label=f'$n = {ns[i]}$', color = f'{colors[i]}', linewidth=0.5)

  for i in range(3):
    fixed_t2e_means = fixed_t2e_means_list[i]
    fixed_t2e_stds = fixed_t2e_stds_list[i]
    plt.errorbar(eces, fixed_t2e_means, yerr=fixed_t2e_stds, color = f'{colors[i]}', linestyle='dotted', linewidth=0.5)

  for i in range(3):
    logistic_t2e_means = logistic_t2e_means_list[i]
    logistic_t2e_stds = logistic_t2e_stds_list[i]
    plt.errorbar(eces, logistic_t2e_means, yerr=logistic_t2e_stds, color = f'{colors[i]}', linestyle='dashdot', linewidth=0.5)

  plt.xlabel('$\ell_2$-ECE')
  plt.ylabel('Type II error')
  plt.axhline(y=0.95, color='k', linestyle='--', linewidth=0.5)
  line1 = Line2D([0], [0], label='optimal bin', color='k', linewidth=0.5)
  line2 = Line2D([0], [0], label='fixed bin', color='k', linestyle='dotted', linewidth=0.5)
  line3 = Line2D([0], [0], label='logistic', color='k', linestyle='dashdot', linewidth=0.5)
  handles, _ = plt.gca().get_legend_handles_labels()
  handles.extend([line1, line2, line3])
  plt.legend(handles=handles, framealpha=0.5, loc='lower left', prop={'size': 6})
  plt.savefig('figures/compare.pgf', bbox_inches = 'tight')
