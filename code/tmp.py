from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from functools import partial
from datasets import prepare_data
from fastai.tabular.data import TabularList
from fastai.tabular.transform import FillMissing, Normalize, Categorify
from fastai.metrics import accuracy
from fastai_ext.augmentations import TabularMixUpCallback, SwapNoiseCallback
from fastai_ext.model import tabular_learner, dae_learner
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs, display_embs, display_emb
from fastai_ext.utils import request_lr, transfer_from_dae, freeze_but_last, unfreeze_all
import pdb
from sklearn.decomposition import PCA
import numpy as np

path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)

_, valid_ids = next(kf.split(df))

data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))

xb, yb = next(iter(data.train_dl))
from fastai_ext.augmentations import choose_corruption
choose_corruption(xb[0], 0.15)
inp = xb[0]
nrows, ncols = inp.shape
np.random.choice(ncols, (1,int(ncols*alpha)), replace=False)

learn_dae = dae_learner(data, layers=[100,100], metrics=None, swap_noise=0, ps=[0.5,0.5])
lr_pre = request_lr(learn_dae)
learn_dae.fit_one_cycle(15, lr_pre)
learn_dae.save('dae_tmp')
learn_dae.load('dae_tmp')

display_embs(learn_dae, nrows=None, ncols=3)
plt.show()

learn = tabular_learner(data, layers=[100,100], metrics=accuracy, mixup_alpha=0, ps=[0.01,0.01])

learn.split(lambda m: m.layers[-1])
learn_dae.split(lambda m: m.layers[-1])

learn.layer_groups[0].load_state_dict(learn_dae.layer_groups[0].state_dict())

def transfer_overlap(m1, m2):
    pretrained_dict = m2.state_dict()
    model_dict = m1.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    m1.load_state_dict(model_dict)


transfer_overlap(learn.layer_groups[0], learn_dae.layer_groups[0])

# learn.split(lambda m: m.layers[-1])
transfer_from_dae(learn, learn_dae)
# learn.freeze()
freeze_but_last(learn)

lr_last = request_lr(learn)
learn.fit_one_cycle(1, lr_last)
# learn.unfreeze()
unfreeze_all(learn)

lr = request_lr(learn)
learn.fit_one_cycle(5, lr)
learn.save('clf_tmp')

display_embs(learn, nrows=None, ncols=3)
plt.show()