import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def f1(x):
    return x**2

def f2(x):
    return x**2+100#-x**2 + 40*x

x11 = np.arange(-18,18,0.5)
x12 = f1(x11)

x21 = np.arange(-18,18,0.5)
x22 = f2(x21)

x1 = np.concatenate([x11, x21])
x2 = np.concatenate([x12, x22])

X = np.hstack([x1[:,None], x2[:,None]]).astype(np.float32)
y = np.concatenate([np.zeros_like(x11), np.ones_like(x12)]).astype(np.int64)

X = (X-np.mean(X,0))/np.std(X,0)

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X[y==0, 0], X[y==0,1], label='class 1')
ax.scatter(X[y==1, 0], X[y==1,1], label='class 2')
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.savefig('../writing/figures/simple_dataset.pdf', bbox_inches='tight')

from fastai import *

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

model = SimpleCNN()

ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
dl = DataLoader(dataset=ds, batch_size=8, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

tmp = []
for epoch in range(100):
    train_epoch(model, dl, optimizer, F.cross_entropy)
    tmp += [validate(model, dl, F.cross_entropy)]

X_mins = np.min(X, 0)*1.05
X_maxs = np.max(X, 0)*1.05

x1_all = np.arange(X_mins[0], X_maxs[0], (X_maxs[0]-X_mins[0])/100)
x2_all = np.arange(X_mins[1], X_maxs[1], (X_maxs[1]-X_mins[1])/100)

X_all = np.vstack(list(itertools.product(x1_all,x2_all))).astype(np.float32)

ds_all = TensorDataset(torch.from_numpy(X_all))
dl_all = DataLoader(dataset=ds_all, batch_size=32)

def get_shade_preds(model:nn.Module, dl:DataLoader, sft=True)->None:
    model.eval()
    preds=[]
    for xb in dl:
        preds.append(model(xb[0]))
    preds = torch.cat(preds)
    if sft: preds = F.softmax(preds, 1)
    return np.array(preds)[:,1]

preds = get_shade_preds(model, dl_all)

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X_all[preds<0.5,0],X_all[preds<0.5,1], alpha=0.2, color=plt.cm.tab10(0))
ax.scatter(X_all[preds>0.5,0],X_all[preds>0.5,1], alpha=0.2, color=plt.cm.tab10(1))
ax.scatter(X[y==0,0], X[y==0,1], label='class 1', color=plt.cm.tab10(0))
ax.scatter(X[y==1,0], X[y==1,1], label='class 2', color=plt.cm.tab10(1))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.savefig('../writing/figures/simple_dataset_simpleNN.pdf', bbox_inches='tight')


class ComplexCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.a1 = nn.Sigmoid()
        self.fc2 = nn.Linear(2, 2)
#         self.a1 = nn.ReLU(inplace=True)
#         self.a1 = nn.Tanh()
    
    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.fc2(x)
#         x = F.softmax(x, 1)
        return x

model = ComplexCNN()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

tmp = []
for epoch in range(500):
    train_epoch(model, dl, optimizer, F.cross_entropy)
    tmp += [validate(model, dl, F.cross_entropy)]

preds = get_shade_preds(model, dl_all)

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X_all[preds<0.5,0],X_all[preds<0.5,1], alpha=0.2, color=plt.cm.tab10(0))
ax.scatter(X_all[preds>0.5,0],X_all[preds>0.5,1], alpha=0.2, color=plt.cm.tab10(1))
ax.scatter(X[y==0,0], X[y==0,1], label='class 1', color=plt.cm.tab10(0))
ax.scatter(X[y==1,0], X[y==1,1], label='class 2', color=plt.cm.tab10(1))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.savefig('../writing/figures/simple_dataset_complexNN.pdf', bbox_inches='tight')

extractor=nn.Sequential(*[model.fc1, model.a1])

extractor.eval()
rep=[]
y_new=[]
for xb,yb in dl:
    rep += [extractor(xb)]
    y_new += [yb]
rep = np.array(torch.cat(rep))
y_new = np.array(torch.cat(y_new))

fig, ax = plt.subplots(figsize=(10,6))
# ax.scatter(X_all[preds<0.5,0],X_all[preds<0.5,1], alpha=0.2, color=plt.cm.tab10(0))
# ax.scatter(X_all[preds>0.5,0],X_all[preds>0.5,1], alpha=0.2, color=plt.cm.tab10(1))
ax.scatter(rep[y_new==0,0], rep[y_new==0,1], label='class 1', color=plt.cm.tab10(0))
ax.scatter(rep[y_new==1,0], rep[y_new==1,1], label='class 2', color=plt.cm.tab10(1))
plt.tick_params(which='both', left=False, bottom=False, labelbottom=False, labelleft=False)
ax.set_xlabel('Hidden Dimension 1')
ax.set_ylabel('Hidden Dimension 2')
ax.legend()
plt.savefig('../writing/figures/simple_dataset_complexNN_rep.pdf', bbox_inches='tight')