from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pdb

path = Path('../data/adult')
df = pd.read_csv(path/'adult.csv')

dep_var = '>=50k'
num_vars = ['age', 'hours-per-week', 'capital-gain', 'capital-loss'] #, 'fnlwgt', 'education-num'
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

plt.rcParams.update({'font.size': 20})

fig, ((a,b),(c,d)) = plt.subplots(2,2, figsize=(20,20))

for i, (ax,var) in enumerate(zip([a,b,c,d], num_vars)):
    sns.kdeplot(df.loc[df[dep_var]==1,var], label='>50k', ax=ax)
    sns.kdeplot(df.loc[df[dep_var]==0,var], label='<50k', ax=ax)
    ax.set_xlabel(var)

fig.tight_layout()
plt.savefig('adult_cont.png', bbox_inches='tight')

exit()

for var in cat_vars:
    df[var]=df[var].str.slice(stop=4)

fig, ((a,b),(c,d),(e,f), (g,h)) = plt.subplots(4,2,figsize=(20,30))
sns.countplot(df['workclass'],hue=df[dep_var],ax=f)
sns.countplot(df['relationship'],hue=df[dep_var],ax=b)
sns.countplot(df['marital-status'],hue=df[dep_var],ax=c)
sns.countplot(df['race'],hue=df[dep_var],ax=d)
sns.countplot(df['sex'],hue=df[dep_var],ax=e)
sns.countplot(df['native-country'],hue=df[dep_var],ax=a)
a.set_xticks([])
sns.countplot(df['education'],hue=df[dep_var],ax=g)
sns.countplot(df['occupation'],hue=df[dep_var],ax=h)

for ax in [a,b,c,d,e,f,g,h]:
    ax.tick_params(axis='x', rotation=45)

fig.tight_layout()
plt.savefig('adult_cat.png', bbox_inches='tight')