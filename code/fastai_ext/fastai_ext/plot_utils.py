from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt


def plot_best(summary_df, param_names, metric_names, ylim1=None, ylim2=None):
    mean_df = (summary_df.groupby(['index']).tail(1)
               .pivot(index=param_names[0], columns=param_names[1], 
                      values=[(metric_names[0], 'mean'), (metric_names[1], 'mean')]))
    sigma_df = (summary_df.groupby(['index']).tail(1)
                .pivot(index=param_names[0], columns=param_names[1], 
                       values=[(metric_names[0], 'std'), (metric_names[1], 'std')]))

    fig, ax = plt.subplots(1, len(metric_names), figsize=(20,6))
    for i, met in enumerate(metric_names):
        mu = mean_df[((met, 'mean'),)]
        # mu.index = ['False', 'True']
        sigma = sigma_df[((met, 'std'),)]
        # sigma.index = ['False', 'True']
        for j in range(mu.shape[1]):
            ax[i].plot(mu.iloc[:,j], label=mu.columns[j])
            ax[i].fill_between(mu.index, mu.iloc[:,j]-sigma.iloc[:,j], mu.iloc[:,j]+sigma.iloc[:,j], alpha=0.3)
            ax[i].set_ylabel(met)
            ax[i].set_xlabel(mu.index.name)
            ax[i].legend(title=param_names[1])
            ax[i].set_xticks(mu.index)
    if ylim1: ax[0].set_ylim(ylim1)
    if ylim2: ax[1].set_ylim(ylim2)


def plot_over_epochs(summary_df, param_names, metric_names, config_df, ylim1=None, ylim2=None):
    mean_df = summary_df.pivot(index='epoch', columns='index', 
                               values=[(metric_names[0], 'mean'), (metric_names[1], 'mean')])
    sigma_df = summary_df.pivot(index='epoch', columns='index', 
                                values=[(metric_names[0], 'std'), (metric_names[1], 'std')])

    fig, ax = plt.subplots(1, len(metric_names), figsize=(20,6))
    for i, met in enumerate(metric_names):
        mu = mean_df[((met, 'mean'),)]
        sigma = sigma_df[((met, 'std'),)]
        for j in range(mu.shape[1]):
            ax[i].plot(mu.iloc[:,j], label=f"({','.join([str(p) for p in config_df.loc[mu.columns[j]]])})")
            ax[i].fill_between(mu.index, mu.iloc[:,j]-sigma.iloc[:,j], mu.iloc[:,j]+sigma.iloc[:,j], alpha=0.3)
            ax[i].set_ylabel(met)
            ax[i].set_xlabel(mu.index.name)
            ax[i].legend(title=f"({','.join(param_names)})")
            ax[i].set_xticks(mu.index)
    if ylim1: ax[0].set_ylim(ylim1)
    if ylim2: ax[1].set_ylim(ylim2)

def display_emb(learn, cat_idx, ax=None, **kwargs):
    cat_name = learn.data.cat_names[cat_idx]
    cat_classes = learn.data.classes[cat_name]
    W = learn.model.embeds[cat_idx].weight.detach().cpu().numpy()

    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(W)

    if ax == None: fig, ax = plt.subplots(**kwargs)
    ax.scatter(pca_out[:,0], pca_out[:,1])
    for i, txt in enumerate(cat_classes):
        ax.annotate(txt, (pca_out[i,0], pca_out[i,1]))
    ax.set_title(cat_name)
    ax.text(1,1, np.sum(pca.explained_variance_ratio_), transform = ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom')

def display_embs(learn, nrows, ncols=None, **kwargs):
    ncats = len(learn.data.cat_names)
    if nrows: 
        ncols = ncats//nrows
        if ncats%nrows > 0: ncols += 1
    elif ncols: 
        nrows = ncats//ncols
        if ncats%ncols > 0: nrows += 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,20))
    for i in range(ncats):
        display_emb(learn, i, ax = ax[i//ncols, i%ncols])
