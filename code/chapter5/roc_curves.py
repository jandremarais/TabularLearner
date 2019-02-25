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
import pandas as pd
from matplotlib import pyplot as plt
from fastai_ext.utils import request_lr, auto_lr
from fastai_ext.hyperparameter import create_experiment, record_experiment, get_config_df, summarise_results, load_results
from fastai_ext.plot_utils import plot_best, plot_over_epochs, display_embs
from fastai_ext.model import tabular_learner
import pdb
from sklearn.metrics import roc_curve, auc
from scipy import interp

path = Path('../data/adult')
df, dep_var, num_vars, cat_vars = prepare_data(path)

procs = [FillMissing, Categorify, Normalize]

src = TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=num_vars, procs=procs)
kf = KFold(5, random_state=42, shuffle=True)
# _,valid_ids = next(kf.split(df))

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_ids, valid_ids) in enumerate(kf.split(df)):
        data = (src.split_by_idx(valid_ids).label_from_df(cols=dep_var).databunch(bs=512))
        learn = tabular_learner(data, layers=[512,512,512], ps=[0.3,0.3,0.3], metrics=accuracy)
        lr = auto_lr(learn, wd=1e-5)
        learn.fit_one_cycle(5, lr, wd=1e-5)

        preds = learn.get_preds()
        preds = to_np(preds[0][:,1])

        fpr, tpr, thresholds = roc_curve(data.valid_ds.y.items, preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

plt.figure(figsize=(10,6))

plt.rcParams.update({'font.size': 12})

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()

plt.savefig('../writing/figures/roc_curves.pdf', bbox_inches='tight')