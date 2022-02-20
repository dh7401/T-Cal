import sys 
sys.path.append('..')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import ece, perturb_scores, chi_squared_statistic, rejection_sampling

# reproducibility
np.random.seed(42)

n = 20000
s = .6
rho = 100
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
ece_list = [0.0098 * rho * m**(-s) for m in range(40, 250, 10)]

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


n = 20000
s = .6
rho = 100
m_star = int(n**((2/(4*s + 1))))

two_null_stats = []
for _ in range(1000):
  null_z = np.random.uniform(0, 1, n)
  null_y = np.random.binomial(1, null_z, n)
  null_sample1, null_sample2 = rejection_sampling(null_z, null_y)
  two_null_stats.append(chi_squared_statistic(null_sample1, null_sample2, m_star))
two_null_stats.sort()
threshold = two_null_stats[-50]

two_t2e_mean_list = []
two_t2e_std_list = []

for m in range(40, 250, 10):
  t2e = []
  for _ in range(3):
    two_alt_stats = []
    for _ in range(1000):
      null_z = np.random.uniform(0, 1, n)
      alt_z = perturb_scores(null_z, m, s, (-1) ** np.floor(2 * m * (null_z - 0.25)), rho)
      alt_y = np.random.binomial(1, alt_z, n)
      alt_sample1, alt_sample2 = rejection_sampling(null_z, alt_y)
      two_alt_stats.append(chi_squared_statistic(alt_sample1, alt_sample2, m_star))
    two_alt_stats.sort()
    t2e.append(np.searchsorted(two_alt_stats, threshold) / 1000)
  two_t2e_mean_list.append(np.mean(t2e))
  two_t2e_std_list.append(np.std(t2e))


plt.figure(figsize=(2.5, 2))
plt.xlabel('$\ell_2$-ECE')
plt.ylabel('Type II error')

plt.errorbar(ece_list, one_t2e_mean_list, yerr=one_t2e_std_list, label='debiased plug-in', color='b', linewidth=0.5)
plt.errorbar(ece_list, two_t2e_mean_list, yerr=two_t2e_std_list, label='sample splitting', color='r', linewidth=0.5)

plt.axhline(y=0.95, color='k', linestyle='--')
plt.legend(framealpha=0.5, loc='lower left')
plt.savefig('figures/type2.pgf', bbox_inches = "tight", prop={'size': 6})
