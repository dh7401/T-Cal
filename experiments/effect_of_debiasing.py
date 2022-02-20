import sys 
sys.path.append('..')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import ece, perturb_scores

# reproducibility
np.random.seed(42)


s = .3
n = 10000
m = int(n**(2 / (4* s + 1)))

ece2b_null = []
ece2b_alt = []

ece2d_null = []
ece2d_alt = []

ece1_null = []
ece1_alt = []

for _ in range(1000):
  null_z = np.random.uniform(size=n)
  null_y = np.random.binomial(1, null_z, n)
  alt_z = perturb_scores(null_z, m//2, s, 1 - 2*(null_z <= 0.5), 100)
  alt_y = np.random.binomial(1, alt_z, n)

  ece2b_null.append(ece(null_z, null_y, m))
  ece2b_alt.append(ece(null_z, alt_y, m))

  ece2d_null.append(ece(null_z, null_y, m, debias=True))
  ece2d_alt.append(ece(null_z, alt_y, m, debias=True))

  ece1_null.append(ece(null_z, null_y, m, 1))
  ece1_alt.append(ece(null_z, alt_y, m, 1))


if __name__ == '__main__':
    # pyplot setup
    plt.rcParams['pgf.preamble'] = r'\usepackage{amsfonts}'
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams['pgf.rcfonts'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = '9'
    plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(1.7, 1.7))
    plt.hist(ece2b_null, label='$P_0$', alpha=0.5, bins=20, color='b')
    plt.hist(ece2b_alt, label='$P_1$', alpha=0.5, bins=20, color='r')
    plt.axvline(np.mean(ece2b_null), color='b', linestyle='dashed')
    plt.axvline(np.mean(ece2b_alt), color='r', linestyle='dashed')
    plt.xlabel('$T_{m, n}^{\mathrm{b}}$')
    plt.ylabel('count')
    plt.legend(framealpha=0.5)
    plt.savefig('figures/bias-03.pgf', bbox_inches = "tight")

