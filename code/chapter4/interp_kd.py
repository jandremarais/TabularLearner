from eli5.permutation_importance import get_score_importances
from sklearn.metrics import log_loss
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
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import xgboost as xgb

path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)
_,valid_ids = next(kf.split(df))

data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))
learn = tabular_learner(data, layers=[200,200], metrics=accuracy)
wd=1e-5
# lr = request_lr(learn, wd=wd)
# learn.fit_one_cycle(10, lr, wd=wd)
# learn.save('tmp')
learn.load('tmp')

def nn_predict_from_df(X):
    bs=512
    m=learn.model.eval()
    n_cats=len(learn.data.valid_ds.cat_names)
    X_cats = tensor(X[:,:n_cats].astype(np.int64))
    X_conts= tensor(X[:, n_cats:]).float()
    preds=[]
    for ind in np.arange(0, len(X), bs):
        x_cats_b = X_cats[ind:(ind+bs),:]
        x_conts_b = X_conts[ind:(ind+bs),:]
        pred = m(x_cats_b, x_conts_b)
        pred = F.softmax(pred, dim=1)
        pred = pred.detach().numpy()
        pred = pred[:,1]
        preds.append(pred)
    return np.concatenate(preds)

train_x = np.concatenate([learn.data.train_ds.codes, learn.data.train_ds.conts],1).astype(np.float32)
val_x = np.concatenate([learn.data.valid_ds.codes, learn.data.valid_ds.conts],1).astype(np.float32)

train_y = nn_predict_from_df(train_x)
val_y = nn_predict_from_df(val_x)

train_x = pd.DataFrame(train_x)
train_x.columns = learn.data.train_ds.cat_names+learn.data.train_ds.cont_names

val_x = pd.DataFrame(val_x)
val_x.columns = learn.data.valid_ds.cat_names+learn.data.valid_ds.cont_names

# feature_names = learn.data.train_ds.cat_names+learn.data.train_ds.cont_names
# dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
# dvalid = xgb.DMatrix(val_x, label=val_y, feature_names=feature_names)

model = XGBRegressor(objective='binary:logistic', n_estimators=500, max_depth=10)


model.fit(train_x, train_y, verbose=True, eval_set=[(val_x, val_y)])
plot_importance(model, xlabel="Importance", ylabel='column')
plt.savefig('../writing/figures/xgb_importance.png', bbox_inches='tight')
plt.show()