from sklearn.datasets import make_moons, make_circles
from matplotlib import pyplot as plt
import pdb
import numpy as np
from sklearn.model_selection import train_test_split
from fastai.basic_data import TensorDataset, DataBunch, DatasetType
from fastai.core import Path
from fastai.basic_train import Learner
import torch.nn.functional as F
from fastai.torch_core import nn, torch
from fastai_ext.utils import request_lr, StoreHook
import itertools
from mpl_toolkits.mplot3d import Axes3D


path = Path('../../data')

# X, y = make_moons(n_samples=200, shuffle=True, noise=0.05, random_state=42)
X, _ = make_circles(n_samples=400, shuffle=False, noise=0.05, random_state=42, factor=0.5)
X = X[:200,:]
X = X.astype(np.float32)
y = np.arange(len(X))/len(X)
y = y.astype(np.float32)

np.random.seed(42)
shuff_ids = np.random.choice(len(X),len(X), replace=False)
X = X[shuff_ids,:]
y = y[shuff_ids]

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

X_mins = np.min(X, 0)
X_maxs = np.max(X, 0)
x1 = np.arange(X_mins[0], X_maxs[0], step=(X_maxs[0]-X_mins[0])/100)
x2 = np.arange(X_mins[1], X_maxs[1], step=(X_maxs[1]-X_mins[1])/100)
X_bg = np.vstack(list(itertools.product(x1, x2))).astype(np.float32)
y_bg = np.zeros(X_bg.shape[0])

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
test_ds = TensorDataset(torch.from_numpy(X_bg), torch.from_numpy(y_bg))

data = DataBunch.create(train_ds, valid_ds, test_ds, path=path, bs=32)

class simpleNN(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = []
        for i, (n1, n2) in enumerate(zip(sizes[:-2], sizes[1:-1])):
            self.layers += [nn.Linear(n1, n2), nn.Tanh(), nn.BatchNorm1d(n2)] #ReLU(inplace=True)
        
        self.layers = nn.Sequential(*self.layers)
        self.head = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        x = self.layers(x)
        x = self.head(x)
        x = F.sigmoid(x)
        return x.view(-1)

model = simpleNN(sizes=[2,10,2,1])

learn = Learner(data, model, loss_func=F.mse_loss)

lr = request_lr(learn)

learn.fit(20, lr)

preds_bg, _ = learn.get_preds(DatasetType.Test)
preds_bg = preds_bg.cpu().numpy()

learn.callbacks += [StoreHook(learn.model.layers[-1])]
_, y_true = learn.get_preds(DatasetType.Train)
y_true = y_true.cpu().numpy()
cb = learn.callbacks[0]
hidden = torch.cat(cb.outputs).cpu().numpy()

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X_bg[:,0], X_bg[:,1], color=plt.cm.viridis(preds_bg), alpha=0.3)
ax.scatter(X[:,0], X[:,1], color=plt.cm.viridis(np.clip(y, 0.001, 0.999)))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(hidden[:,0], hidden[:,1], c=y_true)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()
