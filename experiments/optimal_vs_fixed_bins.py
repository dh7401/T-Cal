import sys 
sys.path.append('..')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import ece, perturb_scores, chi_squared_statistic, rejection_sampling

# reproducibility
np.random.seed(42)

np.random.seed(42)

s = .6
rho = 100
B = 15

fix_t2e_mean_ns = []
fix_t2e_std_ns = []


for n in [2000, 5000, 10000]:

  one_null_stats = []
  for _ in range(1000):
      null_z = np.random.uniform(0, 1, n)
      null_y = np.random.binomial(1, null_z, n)
      one_null_stats.append(ece(null_z, null_y, B, debias=True))
  one_null_stats.sort()
  threshold = one_null_stats[-50]

  fix_t2e_mean_list = []
  fix_t2e_std_list = []

  for m in range(40, 250, 10):
    t2e = []
    for _ in range(3):
      one_alt_stats = []
      for _ in range(1000):
        null_z = np.random.uniform(0, 1, n)
        alt_z = perturb_scores(null_z, m, s, (-1) ** np.floor(2 * m * (null_z - 0.25)), rho)
        alt_y = np.random.binomial(1, alt_z, n)
        one_alt_stats.append(ece(null_z, alt_y, B, debias=True))
      one_alt_stats.sort()
      t2e.append(np.searchsorted(one_alt_stats, threshold) / 1000)
    fix_t2e_mean_list.append(np.mean(t2e))
    fix_t2e_std_list.append(np.std(t2e))
  
  fix_t2e_mean_ns.append(fix_t2e_mean_list)
  fix_t2e_std_ns.append(fix_t2e_std_list)
  
  one_t2e_mean_ns = []
one_t2e_std_ns = []


for n in [2000, 5000, 10000]:
  m_star = int(n**((2/(4*s + 1))))
  one_null_stats = []
  for _ in range(1000):
      null_z = np.random.uniform(0, 1, n)
      null_y = np.random.binomial(1, null_z, n)
      one_null_stats.append(ece(null_z, null_y, m_star, debias=True))
  one_null_stats.sort()
  threshold = one_null_stats[-50]

  one_t2e_mean_list = []
  one_t2e_std_list = []

  for m in range(40, 250, 10):
    t2e = []
    for _ in range(3):
      one_alt_stats = []
      for _ in range(1000):
        null_z = np.random.uniform(0, 1, n)
        alt_z = perturb_scores(null_z, m, s, (-1) ** np.floor(2 * m * (null_z - 0.25)), rho)
        alt_y = np.random.binomial(1, alt_z, n)
        one_alt_stats.append(ece(null_z, alt_y, m_star, debias=True))
      one_alt_stats.sort()
      t2e.append(np.searchsorted(one_alt_stats, threshold) / 1000)
    one_t2e_mean_list.append(np.mean(t2e))
    one_t2e_std_list.append(np.std(t2e))
  
  one_t2e_mean_ns.append(one_t2e_mean_list)
  one_t2e_std_ns.append(one_t2e_std_list)

  from matplotlib.lines import Line2D

plt.figure(figsize=(4, 2))
plt.xlabel('$\ell_2$-ECE')
plt.ylabel('Type II error')
ece_list = [0.0098 * rho * m**(-s) for m in range(40, 250, 10)]

for i in range(3):
  ns = [2000, 5000, 10000]
  colors = ['r', 'g', 'b']
  fix_t2e_mean_list = fix_t2e_mean_ns[i]
  fix_t2e_std_list = fix_t2e_std_ns[i]
  plt.errorbar(ece_list, fix_t2e_mean_list, yerr=fix_t2e_std_list, color = f'{colors[i]}', linewidth=0.5, linestyle='dotted')
for i in range(3):
  one_t2e_mean_list = one_t2e_mean_ns[i]
  one_t2e_std_list = one_t2e_std_ns[i]
  plt.errorbar(ece_list, one_t2e_mean_list, yerr=one_t2e_std_list, label=f'$n = {ns[i]}$', color = f'{colors[i]}', linewidth=0.5)

plt.axhline(y=0.95, color='k', linestyle='--')
line1 = Line2D([0], [0], label='fixed', color='k', linestyle='dotted')
line2 = Line2D([0], [0], label='optimal', color='k')
handles, labels = plt.gca().get_legend_handles_labels()
handles.extend([line1, line2])
plt.legend(handles=handles, framealpha=0.5, loc='lower left', prop={'size': 6})
plt.savefig('figures/compare.pgf', bbox_inches = "tight")
