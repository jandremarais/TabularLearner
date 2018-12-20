from pathlib import Path
from sklearn.model_selection import KFold
from functools import partial
from matplotlib import pyplot as plt
from datasets import prepare_data
from fastai.tabular.data import TabularList
from fastai.tabular.transform import FillMissing, Normalize, Categorify
from fastai.metrics import accuracy
from fastai_ext.model import tabular_learner, dae_learner
from fastai_ext.augmentations import TabularMixUpCallback, SwapNoiseCallback
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs
from fastai_ext.utils import request_lr, transfer_from_dae, freeze_but_last, unfreeze_all
import pdb


path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

experiment_name, exp_path = create_experiment('pretraining', path)

config={'pretrain':[False, True]}
config_df = get_config_df(config)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)

config_df.to_csv(exp_path/'config.csv')
for i, params in config_df.iterrows():
    for fold, (train_ids, valid_ids) in enumerate(kf.split(df)):
        data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))

        if params['pretrain']: 
                learn_dae = dae_learner(data, layers=[500,250], metrics=None, swap_noise=0.15)
                if fold == 0: lr_pre = request_lr(learn_dae)
                learn_dae.fit_one_cycle(15, lr_pre)

        learn = tabular_learner(data, layers=[500,250], metrics=accuracy, mixup_alpha=0)

        if params['pretrain']: 
                # learn.split(lambda m: m.layers[-1])
                transfer_from_dae(learn, learn_dae)
                # learn.freeze()
                freeze_but_last(learn)
                if fold == 0: lr_last = request_lr(learn)
                learn.fit_one_cycle(1, lr_last)
                # learn.unfreeze()
                unfreeze_all(learn)

        if fold == 0: lr = request_lr(learn)
        record_experiment(learn, f'{i}-fold_{fold+1}', exp_path.relative_to(path))
        learn.fit_one_cycle(5, lr)

config_df, recorder_df, param_names, metric_names = load_results(exp_path)
summary_df = summarise_results(recorder_df, param_names, metric_names)

# plot_best(summary_df, param_names, metric_names)
# plt.savefig(exp_path/'best.png', bbox_inches='tight')

plot_over_epochs(summary_df, param_names, metric_names, config_df)
plt.savefig(exp_path/'all_epochs.png', bbox_inches='tight')
