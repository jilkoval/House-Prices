"""
Public kernels I have found particularly useful:

* Pedro Marcelino, Comprehensive data exploration with Python
    https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 

to do:
* outliers

to improve:
* more sophisticated estimate of missing values in the test set (MSZoning, GarageCars, GarageArea)
* include ['Neighborhood', 'Condition1', 'Condition2']
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
    plt.clf()
    plt.close()
    
            
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
    ds = ds.copy()
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

def transform_qual_features(df, features):
    for fi in features:        
        name_num = fi+'_num'
        if fi in ['PoolQC', 'FireplaceQu','GarageQual','GarageCond','BsmtQual','BsmtCond']:
            df[name_num] = transform_6scale_nanstr_int(df[fi])
        elif fi in ['BsmtExposure']:
            df[name_num] = transform_5scale_nanstr_int(df[fi])
        elif fi in ['BsmtFinType1','BsmtFinType2']:
            df[name_num] = transform_7scale_nanstr_int(df[fi])
        elif fi in ['KitchenQual','ExterQual', 'ExterCond']:
            if df[fi].isnull().any():
                print("!!!", fi, "NaN present -- not converting !!!" )
            else:
                df[name_num] = transform_5scale_str_int(df[fi])
    return df
    
def transform_nan_to_0str(df,features):
    df[features] = df[features].fillna('0')
    return df
    
def transform_nan_to_0float(df,features):
    df[features] = df[features].fillna(0.)
    return df
    
def plot_hist_price_per_categories(df,feature,fout=None):
    #df_ = pd.concat([df['SalePrice'], pd.get_dummies(df[feature], columns=[feature])], axis=1)
    grid = sns.FacetGrid(df, col=feature, sharey=False)
    grid.map(plt.hist, 'SalePrice')
    plot_output(plt,fout)
    
def plot_boxplot_categories_vs_price(df,feature,fout=None):
    sns.boxplot(x=feature, y="SalePrice", data=df)
    plot_output(plt,fout)

    
def analyze_cat_feature(df,df2,feature,fout=None):
    print('\n * ', feature)
    print(pd.concat([df[feature],df2[feature]]).value_counts())
    
    if fout is not None:
        f_hist = fout+'_hist_price_per_categories.png'
        f_box = fout+'boxplot_categories_vs_price.png'
    else:
        f_hist = None
        f_box = None
    
    plot_hist_price_per_categories(df,feature,f_hist)
    plot_boxplot_categories_vs_price(df,feature,f_box)
    
def add_dummies(df,features):
    df = pd.concat([df, pd.get_dummies(df[features], columns=[features], prefix=features)], axis=1)
    return df
    
def explore_dummies_corr(df,feature,verbose=True):
    df_with_dum = add_dummies(df,feature)
    cols = [fi for fi in df_with_dum.columns if feature+'_' in fi]
    corrs = df_with_dum.corr(method='spearman')
    if verbose:
        print(corrs.loc['SalePrice', cols])
    return corrs.loc['SalePrice', cols].abs().max()
    

if __name__ in('__main__','__plot__'):
    
    train_raw, test_raw = read_data()
    print('* Data frames: \n\t train:', train_raw.shape, '\n\t test:', test_raw.shape)
    #~ data_overview([train_raw, test_raw])
    
    train, test = make_copies(train_raw, test_raw)
    dfs = [train, test]
    
    ########################
    ### GENERAL OVERVIEW ###
    
    #~ missing_values_overview([train])
    
    #~ plot_correlation_heatmap(train, fout="corr_map_all_num.png")
    #~ plot_correlation_heatmap(train, fout="corr_map_all_num_spearman.png", method='spearman')
    
    ########################################
    ### MIISSING VALUES IN THE TRAIN SET ###
    
    features_not_used_due_to_na = []
    
    # features typically chracterizing a house element type
    # relpace NaN by '0' = house does not have the given element
    # these variables probably need to be converted to dummies
    features_nan_to_0str = ['MiscFeature','Alley','Fence','GarageFinish','GarageType','MasVnrType']
    train = transform_nan_to_0str(train, features_nan_to_0str)
    
    # replace NaN with Float = feature actually has 'zero value'
    features_nan_to_0float = ['LotFrontage','MasVnrArea']
    train = transform_nan_to_0float(train, features_nan_to_0float)
    
    # features that map a quality of certain house element
    # transform to integer scale 0--4, 0--5, 1--5 (0 for NaN -- no element in the house)
    # ? should these be converted to dummies
    features_qual_to_int = ['PoolQC', 'FireplaceQu', 'KitchenQual',
                            'GarageQual','GarageCond',
                            'BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','BsmtFinType2']
    train = transform_qual_features(train, features_qual_to_int)
    
    #~ print(train.loc[:19,['SalePrice','FireplaceQu_num','PoolQC_num']])
    #~ print(train[['SalePrice',
                #~ 'FireplaceQu_num','PoolQC_num','LotFrontage','MasVnrArea',
                #~ 'GarageQual_num','GarageCond_num',
                #~ 'BsmtQual_num','BsmtCond_num','BsmtExposure_num', 'BsmtFinType1_num','BsmtFinType2_num']].corr())
                
    ### GarageYrBlt
    # Missing for the same number of examples as GarageQual etc, i.e., for houses without a garage.
    #~ print(train.corr(method='spearman')[['GarageYrBlt']])
    # Highly correlated with YearBlt and will not be used.
    features_not_used_due_to_na.append('GarageYrBlt')
    
    
    ### Electrical
    #~ analyze_cat_feature(train,test,'Electrical')
    
    # One missing value of 'Electrical'
    # => vast majority of houses seem to have Electrical='SBrkr' => replace NaN by 'SBrkr'
    train['Electrical'] = train['Electrical'].fillna('SBrkr')
    
    #~ data_overview([train])
    #~ plot_correlation_heatmap(train, method='spearman')
    
    #######################################
    ### MISSING VALUES IN THE TEST SET ###
    
    test = transform_nan_to_0str(test, features_nan_to_0str)
    test = transform_nan_to_0float(test, features_nan_to_0float)
    test = transform_qual_features(test, features_qual_to_int)
    
    """
    Apart from the above features that have missing values in the train set,
    there are several features with 1--4 missing values in the test set.
    """
    
    ### MSZoning
    #~ analyze_cat_feature(train,test,'MSZoning',fout='MSZoning_train')
    
    # It looks like MSZoning might be important.
    # The category RL is by far the most common, so I will fill the missing values with RL.
    # There might be a better way to do this, perhaps estimate MSZoning from other features.
    test['MSZoning'] = test['MSZoning'].fillna('RL')
    
    ### Functional
    #~ analyze_cat_feature(train,test,'Functional',fout='Functional_train')
    
    # Not sure if Functional will be important.
    # The category Typ far most common, so use it for NaNs
    test['Functional'] = test['Functional'].fillna('Typ')
    
    ### Utilities
    #~ analyze_cat_feature(train,test,'Utilities')
    
    # Only two categories present and only two examples in the minor class.
    # Will not be used for the modeling.
    features_not_used_due_to_na.append('Utilities')
    
    ### BsmtFullBath and BsmtHalfBath
    #~ print(train.corr(method='spearman')[['BsmtFullBath','BsmtHalfBath']])
    
    # corr_coef with SalePrice: 0.225125, -0.012189
    # BsmtFullBath is strongly correlated with other basement features.
    # BsmtHalfBath is not strongly correlated to anythong.
    # Will not be used for the modeling.
    features_not_used_due_to_na.append('BsmtFullBath')
    features_not_used_due_to_na.append('BsmtHalfBath')
    
    ### KitchenQual
    #~ print(train.corr(method='spearman')[['KitchenQual_num']])
    #~ analyze_cat_feature(train,test,'KitchenQual',fout='KitchenQual_train')
    #~ print(test.loc[test['KitchenQual'].isnull(),'OverallQual'])
    
    # KitchenQual seems to be highly correlated to SalePrice as well as other variables
    # (eg OverallQual, YearRemodAdd, YearBuilt).
    # The correlation between KitchenQual_num and SalePrice is 0.67 
    # and 0.66 with OverallQual, which is average (5).
    # Replace missing value with TA (Average/Typical) which 
    # roughly corresponds to OverallQual and is most common.
    test['KitchenQual'] = test['KitchenQual'].fillna('TA')
    test = transform_qual_features(test, 'KitchenQual')
    
    ### Exterior1st and Exterior2nd
    #~ analyze_cat_feature(train,test,'Exterior1st')
    #~ analyze_cat_feature(train,test,'Exterior2nd')
    
    # Features with many (>20) categories.
    # Might be useful, but probably will not use them for the modelling.
    features_not_used_due_to_na.append('Exterior1st')
    features_not_used_due_to_na.append('Exterior2nd')
    
    ### BsmtFinSF1 and BsmtFinSF2
    #print(train.corr(method='spearman')[['BsmtFinSF1','BsmtFinSF2']])
    
    # corr_coef with SalePrice: 0.301871   -0.038806
    # BsmtFinSF is relatively corralated but also with BsmtFinType1_num (~0.8)
    # or BsmtFullBath (~0.67).
    # Will not be used for modelling
    features_not_used_due_to_na.append('BsmtFinSF1')
    features_not_used_due_to_na.append('BsmtFinSF2')
    
    ### GarageCars and GarageArea
    #~ print(train.corr(method='spearman')[['GarageCars','GarageArea']])
    
    # Highly correlated with SalePrice:  0.690711    0.649379 and might be important.
    # This is a tricky example since we also need adjust all the other Garage variables.
    # But it is only one example, so I will just use 0 here (there is space for improvement here).
    test['GarageCars'] = test['GarageCars'].fillna(0)
    test['GarageArea'] = test['GarageArea'].fillna(0.)
    
    ### SaleType
    #~ analyze_cat_feature(train,test,'SaleType')
    
    # Might be important. Replace missing value by WD which is by far the most common.
    test['SaleType'] = test['SaleType'].fillna('WD')
    
    ################
    #~ missing_values_overview([train,test])
    
    print(" * Features not used due to missing values and mutual correlations: \n \t", features_not_used_due_to_na)
    
    # All the important missing values should be covered now.
    
    
    #############################################
    ### FEATURES HIGHLY CORRELATED WITH PRICE ###
    
    print(train.columns)
    
    # convert quality features (that have not been converted yet) to numeric
    qual_features_to_num = ['ExterQual', 'ExterCond', 'HeatingQC', 'GarageQual', 'GarageCond', 'PoolQC']
    train = transform_qual_features(train, qual_features_to_num)
    test = transform_qual_features(test, qual_features_to_num)
    
    # correlation matrix
    corr_train = train.corr(method='spearman')
    print(corr_train['SalePrice'].abs().sort_values(ascending=False))
    
    # pick numerical features correlated more than certain threshold
    corr_lim = 0.30
    con_features_keep = corr_train[corr_train['SalePrice'].abs()>corr_lim].index.tolist()
    # decided not to be used based on the missing values
    con_features_keep.remove('SalePrice')
    for fi in features_not_used_due_to_na:
        try: con_features_keep.remove(fi)
        except: pass
    
    # check correlation of categorical features through dummy variables and keep those
    # for which at least one is correlated more than given threshold
    cat_features_keep = []
    features_cat = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating',
       'CentralAir', 'Electrical', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType','SaleCondition']    
    for fi in features_cat:
        print('* ',fi)
        max_dummy_corr = explore_dummies_corr(train,fi,verbose=False)
        if max_dummy_corr>corr_lim: cat_features_keep.append(fi)
    for fi in features_not_used_due_to_na:
        try: cat_features_keep.remove(fi)
        except: pass
                    
    print("\n * Continuous features with corr >", corr_lim, len(con_features_keep))
    print(con_features_keep)
    print("\n * Categoric features with highest dummy corr >", corr_lim, len(cat_features_keep))
    print(cat_features_keep)
    
    mutual_lim = 0.8
    #~ plot_correlation_heatmap(train, features=con_features_keep, fout=None, method='spearman')
    features_to_drop_due_to_mutual_cor = []
    for fi in con_features_keep:
        corr_keep = corr_train.loc[con_features_keep,con_features_keep].copy()
        if sum((corr_keep[fi]>mutual_lim).values)>1:
            print('--', fi)
            print(corr_keep.loc[corr_keep[fi]>mutual_lim,fi])
            print(corr_train.loc['SalePrice',corr_keep.index[corr_keep[fi]>mutual_lim].tolist()])
            drop_ind = corr_train.loc['SalePrice',corr_keep.index[corr_keep[fi]>mutual_lim].tolist()].idxmin()
            features_to_drop_due_to_mutual_cor.append(drop_ind)
    print(list(set(features_to_drop_due_to_mutual_cor)))
    
    #~ train_keep = train[con_features_keep.apped(cat_features_keep)].copy(deep=True)
    #~ test_keep = test[con_features_keep.apped(cat_features_keep)].copy(deep=True)

    features_car = ['Neighborhood', 'Condition1', 'Condition2',]
    
    ################################################
    ### MUTUAL CORRELATIONS OF SELECTED FEATURES ###
    
    train = pd.get_dummies(train, dummies=cat_features_keep)
    #~ test = pd.get_dummies(test, cat_features_keep)

    
