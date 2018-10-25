from fastai import *          
from fastai.tabular import *  

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# plt.style.use('dark_background')


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

    
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def save_list(l, fp):
    with open(fp, "wb") as fp: 
        pickle.dump(l, fp)

        
def load_list(fp):
    with open(fp, "rb") as fp: 
        return pickle.load(fp)
    
    
def my_def_emb_sz(df, n, sz_dict, constant=None, ratio=None):
    col = df[n]
    n_cat = len(col.cat.categories)+1  # extra cat for NA
    if constant:
        sz = sz_dict.get(n, min(constant, n_cat-1))
    elif ratio:
        sz = sz_dict.get(n, max(1, int(ratio*n_cat)))
    else:
        sz = sz_dict.get(n, min(50, (n_cat//2)+1))
    return sz

    
def gini_tensor(input, target):
    auc_score = roc_auc_score(target.numpy(), input.numpy()[:,1])
    auc_score = tensor(auc_score)
    return 2*auc_score - 1


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start']
    if time: attr = attr + ['Hour', 'Minute']
    for n in attr: df[n] = getattr(fld.dt, n.lower())
    df['Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
def mae(pred, targ):
    return torch.mean(torch.abs(pred-targ))

def log_mae(pred, targ):
    pred_e = torch.exp(pred)
    targ_e = torch.exp(targ)
    return torch.mean(torch.abs(pred_e-targ_e))