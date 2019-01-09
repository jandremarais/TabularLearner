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

X, y = make_moons(n_samples=200, shuffle=True, noise=0.05, random_state=42)
# X, y = make_circles(n_samples=200, shuffle=True, noise=0.05, random_state=42, factor=0.5)
X = X.astype(np.float32)
# y = y.astype(np.float32)

np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

X_mins = np.min(X, 0)
X_maxs = np.max(X, 0)
x1 = np.arange(X_mins[0], X_maxs[0], step=(X_maxs[0]-X_mins[0])/100)
x2 = np.arange(X_mins[1], X_maxs[1], step=(X_maxs[1]-X_mins[1])/100)
X_bg = np.vstack(list(itertools.product(x1, x2))).astype(np.float32)
y_bg = np.zeros(X_bg.shape[0])

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X[:,0], X[:,1], color=plt.cm.viridis(np.clip(y, 0.001, 0.999)))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
test_ds = TensorDataset(torch.from_numpy(X_bg), torch.from_numpy(y_bg))

data = DataBunch.create(train_ds, valid_ds, test_ds, path=path, bs=32)

class simpleNN(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = []
        for i, (n1, n2) in enumerate(zip(sizes[:-2], sizes[1:-1])):
            self.layers += [nn.Linear(n1, n2), nn.Tanh()] #ReLU(inplace=True)
        
        self.layers = nn.Sequential(*self.layers)
        self.head = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        x = self.layers(x)
        x = self.head(x)
        return x

model = simpleNN(sizes=[2,2])
learn = Learner(data, model, loss_func=F.cross_entropy)

# lr = request_lr(learn)
lr = 2e-1
learn.fit(20, lr)

preds_bg, _ = learn.get_preds(DatasetType.Test)
preds_bg = F.softmax(preds_bg, 1)
preds_bg = preds_bg.cpu().numpy()[:,1]

fig, ax = plt.subplots(figsize=(10,6))
# ax.scatter(X_bg[:,0], X_bg[:,1], color=plt.cm.viridis(preds_bg>0.5), alpha=0.3)
ax.scatter(X_bg[preds_bg<0.5,0], X_bg[preds_bg<0.5,1], alpha=0.2, color=plt.cm.viridis(0.001), label='Class 1')
ax.scatter(X_bg[preds_bg>=0.5,0],X_bg[preds_bg>0.5,1], alpha=0.2, color=plt.cm.viridis(0.999), label='Class 2')
ax.scatter(X[:,0], X[:,1], color=plt.cm.viridis(np.clip(y, 0.001, 0.999)))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()

model = simpleNN(sizes=[2,2,2])
learn = Learner(data, model, loss_func=F.cross_entropy)

lr = request_lr(learn)
# lr = 2e-1
learn.fit(20, lr)

preds_bg, _ = learn.get_preds(DatasetType.Test)
preds_bg = F.softmax(preds_bg, 1)
preds_bg = preds_bg.cpu().numpy()[:,1]

fig, ax = plt.subplots(figsize=(10,6))
# ax.scatter(X_bg[:,0], X_bg[:,1], color=plt.cm.viridis(preds_bg>0.5), alpha=0.3)
ax.scatter(X_bg[preds_bg<0.5,0], X_bg[preds_bg<0.5,1], alpha=0.2, color=plt.cm.viridis(0.001), label='Class 1')
ax.scatter(X_bg[preds_bg>=0.5,0],X_bg[preds_bg>0.5,1], alpha=0.2, color=plt.cm.viridis(0.999), label='Class 2')
ax.scatter(X[:,0], X[:,1], color=plt.cm.viridis(np.clip(y, 0.001, 0.999)))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()

learn.callbacks += [StoreHook(learn.model.layers[-1])]
_, y_true = learn.get_preds(DatasetType.Train)
cb = learn.callbacks[0]
hidden = torch.cat(cb.outputs).cpu().numpy()

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X_bg[:,0], X_bg[:,1], color=plt.cm.viridis(preds_bg), alpha=0.3)
ax.scatter(X[:,0], X[:,1], color=plt.cm.viridis(np.clip(y, 0.001, 0.999)))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()

W = learn.model.head.weight.detach().cpu().numpy()[0,:]
b = learn.model.head.bias.detach().cpu().numpy()[0]

x1 = np.linspace(np.min(hidden[:,0]), np.max(hidden[:,0]), 20)
x2 = np.linspace(np.min(hidden[:,1]), np.max(hidden[:,1]), 20)
x1, x2 = np.meshgrid(x1, x2)
x3 = -(b+x1*W[0]+x2*W[1])/W[2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(hidden[:,0], hidden[:,1], hidden[:,2], marker='.', c=y_true)
ax.plot_surface(x1, x2, x3, linewidth=0, edgecolor=None,
                rstride=1, cstride=1, cmap='Wistia', alpha=0.5)
plt.show()