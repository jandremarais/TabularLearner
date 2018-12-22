from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


path = Path('../../data')

df = pd.read_csv(path/'adult/adult.csv')

print(df.head())
print(df.columns)

x1 = df['age']
x2 = df['hours-per-week']

def gaus_norm(x):
    return (x-np.mean(x))/np.std(x)

def power_norm(x):
    return np.log((1+x)/(1+np.median(x)))

fig, ax = plt.subplots(1,3)

sns.jointplot(x1, x2, alpha=0.3)
plt.savefig('../../writing/figures/cont_vars.pdf', bbox_inches='tight')

sns.jointplot(gaus_norm(x1), gaus_norm(x2), alpha=0.3)
plt.savefig('../../writing/figures/gaus_norm.pdf', bbox_inches='tight')

sns.jointplot(power_norm(x1), power_norm(x2), alpha=0.3)
plt.savefig('../../writing/figures/power_norm.pdf', bbox_inches='tight')