import pandas as pd
import itertools
from functools import partial
from fastai.callbacks import CSVLogger

def get_config_df(config):
    df = pd.DataFrame(list(itertools.product(*config.values())), columns=config.keys())
    df.index = [f'model_{i+1}' for i in range(len(df))]
    return df


def create_experiment(nm, path, folder='results'):
    exp_path = (path/folder/nm)
    exp_path.mkdir(exist_ok=True)
    return nm, exp_path

def record_experiment(learn, fn, exp_path):
    learn.callback_fns.append(partial(CSVLogger, filename=exp_path/fn))


def load_results(exp_path):
    config_df = pd.read_csv(exp_path/'config.csv', index_col=0)
    param_names = config_df.columns.values
    recorder_df=[]
    for p in exp_path.ls():
        if p.name.startswith(tuple(config_df.index.values)):
            df = pd.read_csv(p)
            ind_name, fold_name = p.stem.split('-')
            df['index']=ind_name
            df['fold']=int(fold_name.split('_')[-1].split('.')[0])
            recorder_df.append(df)
    recorder_df = pd.concat(recorder_df)
    metric_names = list(set(recorder_df.columns).symmetric_difference(['index', 'epoch', 'train_loss', 'fold']))
    recorder_df = recorder_df.merge(config_df.reset_index())
    return config_df, recorder_df, param_names, metric_names

def summarise_results(recorder_df, param_names, metric_names):
    return (recorder_df.groupby(['index', *param_names, 'epoch'], as_index=False)
            .agg({k:['mean', 'std'] for k in metric_names}))