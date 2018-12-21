from pathlib import Path
from sklearn.model_selection import KFold
from functools import partial
from datasets import prepare_data
from fastai.tabular.data import TabularList
from fastai.tabular.transform import FillMissing, Normalize, Categorify
from fastai.metrics import accuracy
from fastai.torch_core import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torch.utils.data import TensorDataset
import numpy as np
from fastai.layers import embedding
from fastai.basic_train import Learner
from matplotlib import pyplot as plt
from fastai_ext.utils import request_lr
from fastai_ext.plot_utils import display_embs
import pdb


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_feats):
        super(SelfAttention, self).__init__()
        self.fc_keys = [nn.Linear(d_model, d_model) for _ in range(n_feats)]
        self.fc_queries = [nn.Linear(d_model, d_model) for _ in range(n_feats)]
        self.fc_values = [nn.Linear(d_model, d_model) for _ in range(n_feats)]

    def forward(self, x):
        x_keys = torch.stack([fc(x[:,i]) for i,fc in enumerate(self.fc_keys)], 1)
        x_queries = torch.stack([fc(x[:,i]) for i,fc in enumerate(self.fc_queries)], 1)
        x_values = torch.stack([fc(x[:,i]) for i,fc in enumerate(self.fc_values)], 1)
        x = torch.matmul(x_queries, x_keys.transpose(-2,-1))
        p_attn = F.softmax(x, -1)
        x = torch.matmul(p_attn, x_values)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, n_feat):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.fc_res = nn.Linear(d_model, d_model*h)
        self.attns = [SelfAttention(d_model, n_feat) for _ in range(h)]

    def forward(self, x):
        x_res = self.fc_res(x)
        x = torch.cat([attn(x) for attn in self.attns], 2)
        x = x + x_res
        return x

class AttentionModel(nn.Module):
    def __init__(self, emb_szs, n_conts, n_cats, act_func=nn.ReLU(inplace=True), d_model=2, h=3):
        super(AttentionModel, self).__init__()
        self.act_func = act_func
        self.embeds = nn.ModuleList([embedding(ni, d_model) for ni,nf in emb_szs])
        # n_emb = sum(e.embedding_dim for e in self.embeds)
        self.d_model = d_model
        self.n_feat = n_cats+n_conts
        # self.equal_projection = [nn.Linear(e.embedding_dim, self.d_model) for e in self.embeds]
        self.conts_emb = [nn.Linear(1, self.d_model) for _ in range(n_conts)]
        self.m_attn = MultiHeadAttention(h, self.d_model, self.n_feat)
        self.lin1 = nn.Linear(self.d_model*self.n_feat*h, 100)
        self.do = nn.Dropout(0.5)
        # self.bn = nn.BatchNorm1d(100)
        self.lin2 = nn.Linear(100, 2)

    def forward(self, x_cat, x_cont):
        x = [self.act_func(e(x_cat[:,i])) for i,e in enumerate(self.embeds)]
        # x = [proj(x[i]) for i,proj in enumerate(self.equal_projection)]
        x += [self.act_func(e(x_cont[:,i].view(-1,1))) for i,e in enumerate(self.conts_emb)]
        x = torch.stack(x, 1)
        x = self.m_attn(x)
        x = self.act_func(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(self.do(x))
        return x


# emb_szs = [(3,2),(5,4),(4,3)]
# m = AttentionModel(emb_szs)
# out = m(x)


path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)
_,valid_ids = next(kf.split(df))
data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))

x,y = next(iter(data.train_dl))

emb_szs = data.get_emb_szs({})
model = AttentionModel(emb_szs, n_cats=len(data.cat_names), n_conts=len(data.cont_names),
                       act_func=nn.LeakyReLU(inplace=True), d_model=10, h=3)
# pred = model(x[0])
learn = Learner(data, model, metrics=accuracy)

lr = request_lr(learn)

learn.fit_one_cycle(5, lr)
pdb.set_trace()
learn.save('att_tmp')
learn.load('att_tmp')

learn.model.m_attn

pdb.set_trace()

display_embs(learn, nrows=None,ncols=3)
plt.show()