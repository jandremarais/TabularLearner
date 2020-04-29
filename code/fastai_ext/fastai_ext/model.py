from fastai.torch_core import nn, Tensor, torch
from fastai.core import listify, ifnone
from fastai.callbacks.mixup import MixUpLoss
from fastai_ext.augmentations import TabularMixUpCallback, SwapNoiseCallback, AECallback
import math
from fastai.layers import bn_drop_lin, embedding
from functools import partial
from fastai.basic_train import Learner
import torch.nn.functional as F

# __all__ = ['TabularModel', 'DAE', 'tabular_learner', 'dae_learner']

class TabularModel(nn.Module):
    "Basic model for tabular data."

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None,
                 emb_drop=0., y_range=None, use_bn=True, bn_final=False, 
                 act_func='relu', residual=False, mixup_alpha=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        act_dict = {'relu': nn.ReLU(inplace=True), 'selu': nn.SELU(inplace=True)}
        act_func = act_dict[act_func]
        actns = [act_func] * (len(sizes)-2) + [None]
        if get_layer_name(act_func)=='selu':use_bn=False
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)
        if get_layer_name(act_func)=='selu':
            for l in self.layers:
                if get_layer_name(l) == 'Linear':
                    selu_weights_init(l)
        self.residual = residual
        if residual: 
            res_lin = bn_drop_lin(self.n_emb+self.n_cont, sizes[-2], bn=False, p=dp, actn=act_func)
            self.res_lin = nn.Sequential(*res_lin)
        self.mixup_alpha = mixup_alpha

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont) -> Tensor:
        if self.n_emb != 0:
            if self.mixup_alpha>0: 
                x = x_cat
            else:
                x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
                x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        if self.residual: x = self.layers[-1](self.res_lin(x) + self.layers[:-1](x))
        else: x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x
    
def selu_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.5 / math.sqrt(m.weight.numel()))

    elif classname.find('Linear') != -1:
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns

        m.weight.data.normal_(0.0, 1.0 / math.sqrt(fan_in))
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
        
def get_layer_name(m):
    return m.__class__.__name__

class DAE(nn.Module):
    def __init__(self, emb_szs, n_cont, layers, ps=None,
                 emb_drop=0., use_bn=True, bn_final=False, 
                 act_func='relu'):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        n_emb_in = [e.num_embeddings for e in self.embeds]
        out_szs = [n_cont] + n_emb_in
        self.n_emb,self.n_cont,self.out_szs = n_emb,n_cont,out_szs
        sizes = self.get_sizes(layers, sum(out_szs))
        act_dict = {'relu': nn.ReLU(inplace=True), 'selu': nn.SELU(inplace=True)}
        act_func = act_dict[act_func]
        actns = [act_func] * (len(sizes)-2) + [None]
        if get_layer_name(act_func)=='selu':use_bn=False
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)
        if get_layer_name(act_func)=='selu':
            for l in self.layers:
                if get_layer_name(l) == 'Linear':
                    selu_weights_init(l)
    
    def forward(self, x_cat, x_cont) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
            x = self.layers(x)
        return x.split_with_sizes(self.out_szs, -1)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

def mixed_loss(input, target_cat, target_cont, **kwargs):
    input_cont, input_cats = input[0], input[1:]
    loss = F.mse_loss(input_cont, target_cont, **kwargs) 
    loss += sum([F.cross_entropy(input_cat, target_cat[:,i], **kwargs) for i, input_cat in enumerate(input_cats)])
    return loss

def tabular_learner(data, layers, emb_szs=None, metrics=None,
                    ps=None, emb_drop=0., y_range=None, use_bn=True, 
                    act_func='relu', residual=False, mixup_alpha=0, 
                    swap_noise=0, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = TabularModel(emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                         y_range=y_range, use_bn=use_bn, act_func=act_func, residual=residual, mixup_alpha=mixup_alpha)
    l = Learner(data, model, metrics=metrics, **kwargs)
    if mixup_alpha>0: 
        l.loss_func = MixUpLoss(l.loss_func)
        l.callback_fns.append(partial(TabularMixUpCallback, alpha=mixup_alpha))
    if swap_noise>0:
        l.callback_fns.append(partial(SwapNoiseCallback, alpha=swap_noise))
    return l

def dae_learner(data, layers, emb_szs=None, metrics=None,
                ps=None, emb_drop=0, use_bn=True, 
                act_func='relu', swap_noise=0, **kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = DAE(emb_szs, len(data.cont_names), layers=layers, ps=ps, emb_drop=emb_drop,
                use_bn=use_bn, act_func=act_func)
    l = Learner(data, model, metrics=metrics, loss_func=mixed_loss, **kwargs)
    l.callback_fns.append(AECallback)
    if swap_noise>0:
        l.callback_fns.append(partial(SwapNoiseCallback, alpha=swap_noise))
    return l