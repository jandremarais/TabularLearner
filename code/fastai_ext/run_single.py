import fastai
from fastai import URLs
from fastai.datasets import untar_data
from fastai.tabular.data import TabularList
from fastai_ext.model import tabular_learner, dae_learner
from fastai.tabular.transform import FillMissing, Normalize, Categorify
from sklearn.model_selection import KFold
import pandas as pd
import pdb
from fastai.metrics import accuracy
from fastai_ext.augmentations import TabularMixUpCallback, SwapNoiseCallback
from functools import partial
from matplotlib import pyplot as plt
from fastai.data_block import FloatList
from fastai_ext.utils import request_lr, transfer_from_dae, freeze_but_last, unfreeze_all

print(fastai.show_install())

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')

dep_var = '>=50k'
num_vars = ['age', 'fnlwgt', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)

_, valid_ids = next(kf.split(df))

data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))

learn_dae = dae_learner(data, layers=[100,100], metrics=None, swap_noise=0.2)
# lr = request_lr(learn_dae)
pdb.set_trace()
# learn_dae.fit_one_cycle(15, lr)
# learn_dae.save('tmp_dae')
learn_dae.load('tmp_dae')

learn_cls = tabular_learner(data, layers=[100,100], metrics=accuracy, swap_noise=0)

transfer_from_dae(learn_cls, learn_dae)
freeze_but_last(learn_cls)
lr=request_lr(learn_cls)
learn_cls.fit_one_cycle(1, lr)
unfreeze_all(learn_cls)
lr=request_lr(learn_cls)
learn_cls.fit_one_cycle(5, lr)
