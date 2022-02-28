import sys 
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm

from utils import *


def simulate_null_plugin(n, m, p, debias, num_trials):
  plugin_stats = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    null_y = np.random.binomial(1, z, n)
    plugin_stats.append(plugin_ece(z, null_y, m, p, debias))

  return plugin_stats


def simulate_alt_plugin(n, m, s, rho, p, debias, perturb_m, num_trials):
  plugin_stats = []
  
  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    perturbed_z = perturb_scores(z, perturb_m, s, (-1) ** np.floor(2 * perturb_m * (z - 0.25)), rho)
    alt_y = np.random.binomial(1, perturbed_z, n)
    plugin_stats.append(plugin_ece(z, alt_y, m, p, debias))

  return plugin_stats


def simulate_null_logistic(n, num_trials):
  logistic_scores = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    logit = np.log(z / (1 - z)).reshape(n, 1)
    null_y = np.random.binomial(1, z, n)
    model = sm.Logit(null_y, sm.add_constant(logit))
    score = -model.score((0, 1)).T @ np.linalg.inv(model.hessian((0, 1))) @ model.score((0, 1))
    logistic_scores.append(score)

  return logistic_scores


def simulate_alt_logistic(n, s, rho, perturb_m, num_trials):
  logistic_scores = []

  for _ in range(num_trials):
    z = np.random.uniform(size=n)
    logit = np.log(z / (1 - z)).reshape(n, 1)
    perturbed_z = perturb_scores(z, perturb_m, s, (-1) ** np.floor(2 * perturb_m * (z - 0.25)), rho)
    alt_y = np.random.binomial(1, perturbed_z, n)
    model = sm.Logit(alt_y, sm.add_constant(logit))
    score = -model.score((0, 1)).T @ np.linalg.inv(model.hessian((0, 1))) @ model.score((0, 1))
    logistic_scores.append(score)

  return logistic_scores


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
    plugin_null_stats = simulate_null_plugin(n, m_star, 2, True, 1000)
    plugin_null_stats.sort()
    critical_val = plugin_null_stats[-50]

    plugin_t2e_means = []
    plugin_t2e_stds = []

    for m in ms:
      t2e = []
      for _ in range(10):
        plugin_alt_stats = simulate_alt_plugin(n, m_star, s, rho, 2, True, m, 1000)
        plugin_alt_stats.sort()
        t2e.append(np.searchsorted(plugin_alt_stats, critical_val) / 1000)

      plugin_t2e_means.append(np.mean(t2e))
      plugin_t2e_stds.append(np.std(t2e))
    
    plugin_t2e_means_list.append(plugin_t2e_means)
    plugin_t2e_stds_list.append(plugin_t2e_stds)


  # plug-in test with a fixed binning
  n = 10000

  fixed_null_stats = simulate_null_plugin(n, 15, 1, False, 1000)
  fixed_null_stats.sort()
  critical_val = fixed_null_stats[-50]

  fixed_t2e_means = []
  fixed_t2e_stds = []

  for m in ms:
    t2e = []
    for _ in range(10):
      fixed_alt_stats = simulate_alt_plugin(n, 15, s, rho, 1, False, m, 1000)
      fixed_alt_stats.sort()
      t2e.append(np.searchsorted(fixed_alt_stats, critical_val) / 1000)

    fixed_t2e_means.append(np.mean(t2e))
    fixed_t2e_stds.append(np.std(t2e))
  


  # logistic test
  n = 10000

  null_scores = simulate_null_logistic(n, 1000)
  null_scores.sort()

  critical_val = null_scores[-50]

  logistic_t2e_means = []
  logistic_t2e_stds = []

  for m in ms:
    t2e = []
    for _ in range(10):
      alt_scores = simulate_alt_logistic(n, s, rho, m, 1000)
      alt_scores.sort()
      t2e.append(np.searchsorted(alt_scores, critical_val) / 1000)
    logistic_t2e_means.append(np.mean(t2e))
    logistic_t2e_stds.append(np.std(t2e))
    

  # create plot
  pyplot_setup()

  colors = ['r', 'g', 'b']
  plt.figure(figsize=(3.4, 1.7))

  for i in range(3):
    plugin_t2e_means = plugin_t2e_means_list[i]
    plugin_t2e_stds = plugin_t2e_stds_list[i]
    plt.errorbar(eces, plugin_t2e_means, yerr=plugin_t2e_stds, label=f'$n = {ns[i]}$', color = f'{colors[i]}', linewidth=0.5)

  plt.errorbar(eces, fixed_t2e_means, yerr=fixed_t2e_stds, color = 'b', linestyle='dotted', linewidth=0.5)
  plt.errorbar(eces, logistic_t2e_means, yerr=logistic_t2e_stds, color = 'b', linestyle='dashdot', linewidth=0.5)

  plt.xlabel('$\ell_2$-ECE')
  plt.ylabel('Type II error')
  line1 = Line2D([0], [0], label='Ours', color='k', linewidth=0.5)
  line2 = Line2D([0], [0], label='$\widehat{\ell_1\mathrm{-ECE}}$', color='k', linestyle='dotted', linewidth=0.5)
  line3 = Line2D([0], [0], label='Logistic', color='k', linestyle='dashdot', linewidth=0.5)
  handles, _ = plt.gca().get_legend_handles_labels()
  handles.extend([line1, line2, line3])
  plt.legend(handles=handles, framealpha=0.5, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig('figures/compare.pgf', bbox_inches = 'tight')
