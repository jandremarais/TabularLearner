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

print(df.head())

# df.loc[:, num_vars].hist()
# plt.show()

# df.loc[:, cat_vars + [dep_var]].plot.bar(subplots=True)
# df[dep_var].plot.hist()
# plt.show()

for var in cat_vars:
    df[var]=df[var].str.slice(stop=4)

plt.rcParams.update({'font.size': 20})

fig, ((a,b),(c,d),(e,f), (g,h)) = plt.subplots(4,2,figsize=(20,15))
sns.countplot(df['workclass'],hue=df[dep_var],ax=f)
sns.countplot(df['relationship'],hue=df[dep_var],ax=b)
sns.countplot(df['marital-status'],hue=df[dep_var],ax=c)
sns.countplot(df['race'],hue=df[dep_var],ax=d)
sns.countplot(df['sex'],hue=df[dep_var],ax=e)
sns.countplot(df['native-country'],hue=df[dep_var],ax=a)
sns.countplot(df['education'],hue=df[dep_var],ax=g)
sns.countplot(df['occupation'],hue=df[dep_var],ax=h)
# a.set_xticks(rotation=45)
# a.xaxis.set_tick_params(rotation=45)
print('is this still working?')
plt.show()
#plt.savefig('adult_cat2.png')