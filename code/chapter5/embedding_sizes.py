from pathlib import Path
from sklearn.model_selection import KFold
from functools import partial
from datasets import prepare_data
from fastai.tabular.data import TabularList
from fastai.tabular.transform import FillMissing, Normalize, Categorify
from fastai.metrics import accuracy
from fastai.torch_core import torch, to_np
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torch.utils.data import TensorDataset
import numpy as np
from matplotlib import pyplot as plt
from fastai_ext.utils import request_lr, auto_lr
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs, display_embs
from fastai_ext.model import tabular_learner
import pdb

path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

experiment_name, exp_path = create_experiment('embeddingsize', path)

config={'size':[2,4,8]}
config_df = get_config_df(config)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)
# _,valid_ids = next(kf.split(df))

config_df.to_csv(exp_path/'config.csv')
for i, params in config_df.iterrows():
        for fold, (train_ids, valid_ids) in enumerate(kf.split(df)):
                data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))
                emb_szs = {var:params['size'] for var in cat_vars}
                learn = tabular_learner(data, layers=[200,200], metrics=accuracy)
                record_experiment(learn, f'{i}-fold_{fold+1}', exp_path.relative_to(path))
                # if fold==0: lr = auto_lr()#request_lr(learn, wd=1e-5)
                lr = auto_lr(learn)
                learn.fit_one_cycle(5, lr, wd=1e-5)

config_df, recorder_df, param_names, metric_names = load_results(exp_path)
summary_df = summarise_results(recorder_df, param_names, metric_names)

# plot_best(summary_df, param_names, metric_names)
# plt.savefig(exp_path/'best.png', bbox_inches='tight')

plot_over_epochs(summary_df, param_names, metric_names, config_df)
plt.savefig(exp_path/'all_epochs.png', bbox_inches='tight')