import sys 
sys.path.append('..')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import ece, perturb_scores, chi_squared_statistic, rejection_sampling


# reproducibility
np.random.seed(42)

s = .6
rho = 100

logistic_t2e_mean_ns = []
logistic_t2e_std_ns = []

for n in [2000, 5000, 10000]:
  slope_null_stats = []
  intercept_null_stats = []
  for _ in range(1000):
      null_z = np.random.uniform(0, 1, n)
      null_logit = np.log(null_z / (1 - null_z)).reshape(n, 1)
      null_y = np.random.binomial(1, null_z, n)
      clf = LogisticRegression().fit(null_logit, null_y)
      slope_null_stats.append(clf.coef_)
      intercept_null_stats.append(clf.intercept_)
  slope_null_stats.sort()
  intercept_null_stats.sort()

  slope_threshold_l = slope_null_stats[12]
  slope_threshold_r = slope_null_stats[-13]
  intercept_threshold_l = intercept_null_stats[12]
  intercept_threshold_r = intercept_null_stats[-13]


  logistic_t2e_mean_list = []
  logistic_t2e_std_list = []

  for m in range(40, 250, 10):
    t2e = []
    for _ in range(3):
      logistic_alt_stats = []
      for _ in range(1000):
        null_z = np.random.uniform(0, 1, n)
        null_logit = np.log(null_z / (1 - null_z)).reshape(n, 1)
        alt_z = perturb_scores(null_z, m, s, (-1) ** np.floor(2 * m * (null_z - 0.25)), rho)
        alt_y = np.random.binomial(1, alt_z, n)
        clf = LogisticRegression().fit(null_logit, alt_y)
        logistic_alt_stats.append((clf.coef_, clf.intercept_))


      count = 0
      for coef, intercept in logistic_alt_stats:
        if slope_threshold_l <= coef <= slope_threshold_r and intercept_threshold_l <= intercept <= intercept_threshold_r:
          count += 1

      t2e.append(count / 1000)
    logistic_t2e_mean_list.append(np.mean(t2e))
    logistic_t2e_std_list.append(np.std(t2e))
  
  logistic_t2e_mean_ns.append(logistic_t2e_mean_list)
  logistic_t2e_std_ns.append(logistic_t2e_std_list)


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
ns = [2000, 5000, 10000]
colors = ['r', 'g', 'b']

for i in range(3):
  logistic_t2e_mean_list = logistic_t2e_mean_ns[i]
  logistic_t2e_std_list = logistic_t2e_std_ns[i]
  plt.errorbar(ece_list, logistic_t2e_mean_list, yerr=logistic_t2e_std_list, color = f'{colors[i]}', linewidth=0.5, linestyle='dotted')
for i in range(3):
  one_t2e_mean_list = one_t2e_mean_ns[i]
  one_t2e_std_list = one_t2e_std_ns[i]
  plt.errorbar(ece_list, one_t2e_mean_list, yerr=one_t2e_std_list, label=f'$n = {ns[i]}$', color = f'{colors[i]}', linewidth=0.5)

plt.axhline(y=0.95, color='k', linestyle='--')
line1 = Line2D([0], [0], label='slope/intercept', color='k', linestyle='dotted')
line2 = Line2D([0], [0], label='debiased plug-in', color='k')
handles, labels = plt.gca().get_legend_handles_labels()
handles.extend([line1, line2])
plt.legend(handles=handles, framealpha=0.5, loc='lower left', prop={'size': 6})
plt.savefig('figures/slope.pgf', bbox_inches = "tight")
