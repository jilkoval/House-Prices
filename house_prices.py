"""
Public kernels I have found particularly useful:

* Pedro Marcelino, Comprehensive data exploration with Python
    https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
* 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def read_data():
    return pd.read_csv("input/train.csv"), pd.read_csv("input/test.csv")
    
def data_overview(dfs):
    for df_i in dfs:
        print(df_i.columns)
        print(df_i.describe(include = "all"))
        print(df_i.info())
        
def make_copies(df_train, df_test):
    return df_train.copy(deep=True), df_test.copy(deep=True)
    
def missing_values_overview(dfs, labels=['* Train:', '* Test']):
    for i,df_i in enumerate(dfs):
        total = df_i.isnull().sum().sort_values(ascending=False)
        percent = (df_i.isnull().sum()/df_i.shape[0]).sort_values(ascending=False)
        missing_count = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(labels[i], '\n', 
            missing_count[missing_count['Total']>0], '\n')
            
def plot_correlation_heatmap(df, features=None):
    if features is None: corrmat = df.corr()
    else: corrmat = df[features].corr()
    f, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corrmat, vmax=.8, square=True, center=0.)
    plt.show()
    


if __name__ in('__main__','__plot__'):
    
    train_raw, test_raw = read_data()
    print('* Data frames: \n\t train:', train_raw.shape, '\n\t test:', test_raw.shape)
    #data_overview([train_raw, test_raw])
    
    train, test = make_copies(train_raw, test_raw)
    dfs = [train, test]
    
    plot_correlation_heatmap(train)
    #missing_values_overview(dfs)
    
    
    
