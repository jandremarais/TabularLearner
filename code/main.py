from fastai.utils import show_install
import pandas as pd
import numpy as np
from functools import partial
from fastai.tabular import TabularList
from fastai.tabular.transform import FillMissing, Categorify, Normalize
from fastai.tabular.data import tabular_learner
from fastai.metrics import accuracy
from sklearn.model_selection import KFold
from pathlib import Path
from matplotlib import pyplot as plt

from fastai_ext.augmentations import SwapNoiseCallback

print(show_install())

path = Path('data')
df = pd.read_csv(path/'adult.csv')

dep_var = '>=50k'
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
num_vars = ['age', 'fnlwgt', 'education-num', 'capital-gain','capital-loss', 'hours-per-week']

kf = KFold(5, shuffle=True, random_state=42)

procs = [FillMissing, Categorify, Normalize]
src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)

_, valid_ids = next(kf.split(df))

data = src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=128)

learn = tabular_learner(data, layers=[500,500,500], metrics=accuracy, callback_fns=[partial(SwapNoiseCallback, alpha=0.1)])
# learn.callback_fns += [partial(SwapNoiseCallback, alpha=0.1)]

# learn.lr_find()
# learn.recorder.plot()
# plt.show()

lr = 1e-3

learn.fit_one_cycle(10, lr)
# learn.fit(10, lr)