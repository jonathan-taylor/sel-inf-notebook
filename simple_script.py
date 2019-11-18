import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os
import statsmodels.api as sm

curdir = os.path.split(os.path.abspath(__file__))[0]

nsim = 2000
for _ in range(nsim):
    os.system('python %s/simple/inference.py' % curdir)
    pivot = pd.read_csv('%s/simple/results_simple.csv' % curdir)['pivot']
    print(np.mean(pivot), np.std(pivot))
    U = np.linspace(0, 1, 101); plt.clf(); f = plt.figure(figsize=(6,6))
    plt.plot(U, sm.distributions.ECDF(pivot)(U), 'r'); 
    plt.plot([0,1], [0,1], 'k--')
    plt.savefig('%s/simple/results_simple.png' % curdir); plt.close(f)
