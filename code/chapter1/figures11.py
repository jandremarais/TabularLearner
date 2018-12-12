from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import pdb
import numpy as np
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D


X, y = make_classification(n_samples=20, n_features=2, n_redundant=0, random_state=42)

m = LinearRegression()
m.fit(X, y)

x1_range = np.linspace(np.min(X[:,0]), np.max(X[:,1]), num=10)
x2_range = -(m.intercept_ + m.coef_[0]*x1_range - 0.5)/m.coef_[1]

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.plot(x1_range, x2_range, color='black', dashes=[6,2])
ax.set_ylabel('Feature 2')
ax.set_xlabel('Feature 1')
hand = [Line2D([0],[0], marker='o', color=plt.cm.viridis(i+0.1), label=sub, linestyle='None') for i,sub in enumerate(['Class 1', 'Class 2'])]
hand += [Line2D([0],[0],color='black', label='Decision Boundary', dashes=[6,2])]
ax.legend(handles=hand)
# plt.show()
plt.savefig('../writing/figures/linear_boundary.pdf', bbox_inches='tight')