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
import pandas as pd
from matplotlib import pyplot as plt
from fastai_ext.utils import request_lr
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs, display_embs
from fastai_ext.model import tabular_learner
import pdb

path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)
# _,valid_ids = next(kf.split(df))

# data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))
# learn = tabular_learner(data, layers=[200,200], metrics=accuracy)

# plt.rcParams.update({'font.size': 15})

# _, ax = plt.subplots(1,1)
# ax.set_ylabel("Loss")
# ax.set_xlabel("Learning Rate")
# ax.set_xscale('log')
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
# ax.set_xlim(1e-2,0)
# ax.set_ylim(0.45,0.7)

# for wd in [1e-5, 1e-4, 1e-3, 0]:
#     learn.lr_find(wd=wd)
#     lrs = learn.recorder.lrs
#     losses = learn.recorder.losses
#     ax.plot(lrs, losses, label=wd)
# ax.legend(title='Weight Decay')
# # plt.show()
# plt.savefig('../writing/figures/wd_lr_range.pdf', bbox_inches='tight')

# pdb.set_trace()

experiment_name, exp_path = create_experiment('weight_decay', path)

config={'wd':[1e-5, 1e-4, 1e-3, 0]}
config_df = get_config_df(config)
lr=6e-2

config_df.to_csv(exp_path/'config.csv')
for i, params in config_df.iterrows():
    for fold, (train_ids, valid_ids) in enumerate(kf.split(df)):
        data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))

        learn = tabular_learner(data, layers=[200,200], metrics=accuracy)
        record_experiment(learn, f'{i}-fold_{fold+1}', exp_path.relative_to(path))
        
        learn.fit_one_cycle(10, lr, wd=params['wd'])

config_df, recorder_df, param_names, metric_names = load_results(exp_path)
summary_df = summarise_results(recorder_df, param_names, metric_names)

# plot_best(summary_df, param_names, metric_names)
# plt.savefig(exp_path/'best.png', bbox_inches='tight')

plot_over_epochs(summary_df, param_names, metric_names, config_df)
plt.savefig(exp_path/'all_epochs.png', bbox_inches='tight')