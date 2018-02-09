"""
Public kernels I have found particularly useful:

* Pedro Marcelino, Comprehensive data exploration with Python
    https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 

to do:

* missing values -- where are values actually missing, where they correspond to 0
* categorical variables -- check and quantify (dummy) -- 
* deal with NaN in GarageYrBlt
* outliers

X correlation measure -- linear (Pearson) vs. non-linear (Spearman)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="husl")

plots_dir = "./plots/"

def read_data():
    return pd.read_csv("input/train.csv"), pd.read_csv("input/test.csv")
    
def data_overview(dfs):
    for df_i in dfs:
        print(df_i.columns)
        print(df_i.describe(include = "all"))
        print(df_i.info())
        
def make_copies(df_train, df_test):
    return df_train.copy(deep=True), df_test.copy(deep=True)
    
def missing_values_overview(dfs, labels=['* Train:', '* Test'], include_corr=True):
    print("\n*** Missing values counts ***\n")
    for i,df_i in enumerate(dfs):
        total = df_i.isnull().sum().sort_values(ascending=False)
        percent = (df_i.isnull().sum()/df_i.shape[0]).sort_values(ascending=False)
        missing_count = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
        if include_corr and i==0:
            corrmat_per = df_i.corr(method='pearson')['SalePrice']
            corrmat_per = corrmat_per.rename('corr_pearson')
            corrmat_spe = df_i.corr(method='spearman')['SalePrice']
            corrmat_spe = corrmat_spe.rename('corr_spearman')
            missing_count = missing_count.join(corrmat_per)
            missing_count = missing_count.join(corrmat_spe)
        
        print(labels[i], '\n', 
              missing_count[missing_count['Total']>0], '\n')
            
def plot_output(plt_instance,fout=None):
    plt_instance.tight_layout()
    if fout is not None: plt_instance.savefig(plots_dir+fout)
    else: plt_instance.show()
            
def plot_correlation_heatmap(df, features=None, fout=None, method='pearson'):
    
    if features is None: 
        corrmat = df.corr(method=method)
        corrmat = corrmat.drop('Id')
        corrmat = corrmat.drop('Id', axis=1)
    else: corrmat = df[features].corr(method=method)
    
    print("* plotting correlation matrix:", np.shape(corrmat))
    print("\t", corrmat.columns)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, center=0., cmap="BrBG", xticklabels=True, yticklabels=True)
    
    plot_output(plt, fout=fout)

def transform_5scale_str_int(ds):
    ds[ds=='Po'] = 1
    ds[ds=='Fa'] = 2
    ds[ds=='TA'] = 3
    ds[ds=='Gd'] = 4
    ds[ds=='Ex'] = 5 
    return ds.astype('int64')

def transform_6scale_nanstr_int(ds):
    ds = ds.fillna(0)
    ds[ds=='Po'] = 1
    ds[ds=='Fa'] = 2
    ds[ds=='TA'] = 3
    ds[ds=='Gd'] = 4
    ds[ds=='Ex'] = 5
    return ds.astype('int64')
    
def transform_5scale_nanstr_int(ds):
    ds = ds.fillna(0)
    ds[ds=='No'] = 1
    ds[ds=='Mn'] = 2
    ds[ds=='Av'] = 3
    ds[ds=='Gd'] = 4
    return ds.astype('int64')
    
def transform_7scale_nanstr_int(ds):
    ds = ds.fillna(0)
    ds[ds=='Unf'] = 1
    ds[ds=='LwQ'] = 2
    ds[ds=='Rec'] = 3
    ds[ds=='BLQ'] = 4
    ds[ds=='ALQ'] = 5
    ds[ds=='GLQ'] = 6
    return ds.astype('int64')

def transform_cat_features(df, features):
    for fi in features:
        name_num = fi+'_num'
        if fi in ['PoolQC', 'FireplaceQu','GarageQual','GarageCond','BsmtQual','BsmtCond']:
            df[name_num] = transform_6scale_nanstr_int(df[fi])
        elif fi in ['BsmtExposure']:
            df[name_num] = transform_5scale_nanstr_int(df[fi])
        elif fi in ['BsmtFinType1','BsmtFinType2']:
            df[name_num] = transform_7scale_nanstr_int(df[fi])
    return df
    
def transform_nan_to_0str(df,features):
    df[features] = df[features].fillna('0')
    return df
    
def transform_nan_to_0float(df,features):
    df[features] = df[features].fillna(0.)
    return df
    
def plot_dummis_vs_price(df,feature):
    #df_ = pd.concat([df['SalePrice'], pd.get_dummies(df[feature], columns=[feature])], axis=1)
    grid = sns.FacetGrid(df, col=feature, sharey=False)
    grid.map(plt.hist, 'SalePrice')
    plt.show()
    
def replace_electrical_nan(df,rep_str='SBrkr'):
    df['Electrical'] = df['Electrical'].fillna('SBrkr')
    return df

if __name__ in('__main__','__plot__'):
    
    train_raw, test_raw = read_data()
    print('* Data frames: \n\t train:', train_raw.shape, '\n\t test:', test_raw.shape)
    #~ data_overview([train_raw, test_raw])
    
    train, test = make_copies(train_raw, test_raw)
    dfs = [train, test]
    
    #~ missing_values_overview([train])
    
    #~ plot_correlation_heatmap(train, fout="corr_map_all_num.png")
    #~ plot_correlation_heatmap(train, fout="corr_map_all_num_spearman.png", method='spearman')
    
    ### Missing values in train set ###
    
    # features typically chracterizing a house element type
    # relpace NaN by '0' = house does not have the given element
    # these variables probably need to be converted to dummies
    train = transform_nan_to_0str(train, ['MiscFeature','Alley','Fence','GarageFinish','GarageType','MasVnrType'])
    
    # replace NaN with Float = feature actually has 'zero value'
    train = transform_nan_to_0float(train, ['LotFrontage','MasVnrArea'])
    
    # features that map a quality of certain house element
    # transform to integer scale 0--4, 0--5, 1--5 (0 for NaN -- no element in the house)
    # ? should these be converted to dummies
    train = transform_cat_features(train, features=['PoolQC', 'FireplaceQu',
                                                    'GarageQual','GarageCond',
                                                    'BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','BsmtFinType2'])
    
    #~ print(train.loc[:19,['SalePrice','FireplaceQu_num','PoolQC_num']])
    print(train[['SalePrice',
                'FireplaceQu_num','PoolQC_num','LotFrontage','MasVnrArea',
                'GarageQual_num','GarageCond_num',
                'BsmtQual_num','BsmtCond_num','BsmtExposure_num', 'BsmtFinType1_num','BsmtFinType2_num']].corr())
    
    # One missing value of 'Electrical'
    # print(' * Electrical')
    # print('\t',train[['Electrical','SalePrice']].groupby('Electrical').count())
    # => vast majority of houses seem to have Electrical='SBrkr' => replace NaN by 'SBrkr'
    train = replace_electrical_nan(train)
    
    #~ data_overview([train])
    missing_values_overview([train,test])
    #~ plot_correlation_heatmap(train, method='spearman')
    
    ### Missing values in test set ###
    
    
    
    
    
