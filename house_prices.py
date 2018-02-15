"""
Public kernels I have found particularly useful:

* Pedro Marcelino, Comprehensive data exploration with Python
    https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python 
    
* Alexandru Papiu, Regularized Linear Models
    https://www.kaggle.com/apapiu/regularized-linear-models

* massquantity, All You Need is PCA (LB: 0.11421, top 4%)
    https://www.kaggle.com/massquantity/all-you-need-is-pca-lb-0-11421-top-4

* PhilipBall, Journey to the top 10%
    https://www.kaggle.com/fiorenza2/journey-to-the-top-10

to do:
* outliers

to improve:
* more sophisticated estimate of missing values in the test set (MSZoning, GarageCars, GarageArea)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.cross_validation import train_test_split

sns.set(context="notebook", style="whitegrid", palette="husl")
pd.set_option('display.max_rows', 40)
plots_dir = "./plots/"

np.random.seed(seed=666)

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
    
            
def plot_correlation_heatmap(df, features=None, fout=None, method='pearson', verbose=False):
    
    if features is None: 
        corrmat = df.corr(method=method)
        corrmat = corrmat.drop('Id')
        corrmat = corrmat.drop('Id', axis=1)
    else: corrmat = df[features].corr(method=method)
    
    print("* plotting correlation matrix:", np.shape(corrmat))
    if verbose: print("\t", corrmat.columns)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, center=0., cmap="BrBG", xticklabels=True, yticklabels=True)
    
    plot_output(plt, fout=fout)
    
def plot_pairplot(df, features=None, fout=None):
    if features is not None: df_plot = df[features]
    else: df_plot = df
    
    sns.pairplot(df_plot, size = 3)
    
    plot_output(plt, fout=fout)
    
def plot_scatter(df, fx, fy, fout=None):
    sns.lmplot(x=fx, y=fy, data=df)
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
                print("\t !!!", fi, "NaN present -- not converting !!!" )
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
    df = pd.concat([df, pd.get_dummies(df[features], columns=features)], axis=1)
    return df
    
def explore_dummies_corr(df,feature,verbose=True):
    df_with_dum = add_dummies(df,feature)
    cols = [fi for fi in df_with_dum.columns if feature+'_' in fi]
    corrs = df_with_dum.corr(method='spearman')
    if verbose:
        print(corrs.loc['SalePrice', cols])
    return corrs.loc['SalePrice', cols].abs().max()
    
def mutual_corr_more_tha_limit(corr_mat, features, verbose=False):
    correlated_features = []
    corr_keep = corr_mat.loc[features,features].copy()
    for fi in features:
        if sum((corr_keep[fi].abs()>mutual_lim).values)>1:
            correlated_with_fi = corr_keep.loc[corr_keep[fi].abs()>mutual_lim,fi].index.values.tolist()
            f_keep = corr_mat.loc['SalePrice', correlated_with_fi].idxmax()
            correlated_with_fi.remove(f_keep)
            correlated_features.extend(correlated_with_fi)
            if verbose:
                print('\n\t --', fi)
                print(corr_keep.loc[corr_keep[fi].abs()>mutual_lim,fi])
                print(corr_train.loc['SalePrice',corr_keep.index[corr_keep[fi].abs()>mutual_lim].tolist()])
                print('\t remove:', correlated_with_fi)
                print('\t keep:', f_keep)
    return list(set(correlated_features))
    
def scale_data(data_train, data_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data_train)
    return scaler.transform(data_train), scaler.transform(data_test)
    
def scale_and_transform_SalePrice(train_y):
    from sklearn.preprocessing import StandardScaler
    train_y = train_y.apply(lambda x: np.log(x))
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(train_y.values.reshape(-1, 1)).reshape(len(train_y))
    return pd.Series(data=scaled_values, index=train_y.index), scaler
    
def transform_skewed_features(df_train, df_test, lim_skew=0.5):
    from scipy.stats import skew
    features_num = [fi for fi in df_train.columns.tolist() if '_' not in fi]
    skewness = df_train[features_num].apply(lambda x: skew(x))
    skewed_features = skewness[abs(skewness)>lim_skew].index
    print('* Transforming skewed features, skewness>', lim_skew, ':', skewed_features)
    df_train[skewed_features] = np.log1p(df_train[skewed_features])
    df_test[skewed_features] = np.log1p(df_test[skewed_features])
    return df_train, df_test
    
def get_rmse(y,y_true, scaler=None):
    if scaler is not None:
        y =  scaler.inverse_transform(y)
        y_true =  scaler.inverse_transform(y_true)
    return np.sqrt(((y-y_true)**2).sum()/len(y))

def model_LassoCV(X,y,Xtest, scaler=None, verbose=True):
    from sklearn.linear_model import LassoCV
    lasso_1 = LassoCV(alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.], cv=None, max_iter=50000)
    lasso_1.fit(X.values, y)
    alpha = lasso_1.alpha_
    rmse_1 = get_rmse(lasso_1.predict(X.values),y,scaler)
    
    lasso = LassoCV(alphas = np.linspace(0.5,2,10)*alpha, cv=None, max_iter=50000)
    lasso.fit(X.values, y)
    y_hat = lasso.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    
    # coefficicents
    coef = pd.Series(abs(lasso.coef_), index = X.columns)
    features_used = coef[coef != 0].index.tolist()
    
    if verbose:
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        print(coef.sort_values(ascending=False)[:15])
        print('* LassoCV model 1st iteration:')
        print('\t Best alpha:', lasso_1.alpha_)
        print('\t RMSE on train set', rmse_1)
        print('* LassoCV model:')
        print('\t Best alpha:', lasso.alpha_)
        print('\t RMSE on train set', rmse)
    
    return lasso.predict(Xtest), features_used, rmse
    
def run_LassoCV_across_variables(X,y,Xtest,Ytest,features_to_drop=None,scaler=None):
    # run model using all features
    y_test_hat, used_features, rmse_model_base = model_LassoCV(X, y, Xtest, scaler=scaler)
    rmse_base = mean_squared_error(Ytest,y_test_hat)
    print(rmse_model_base, rmse_base, rmse_model_base-rmse_base, len(used_features))
    
    y_test_hat, used_features, rmse_model_base = model_LassoCV(X[used_features], y, Xtest[used_features], scaler=scaler)
    rmse_base = mean_squared_error(Ytest,y_test_hat)
    print(rmse_model_base, rmse_base, rmse_model_base-rmse_base, len(used_features))
    
    # rum models dropping features one by one
    if features_to_drop is None: features_to_drop = used_features
    rmse_list = []
    for fi in features_to_drop:
        X_ = X[used_features].drop(fi, axis=1)
        Xtest_ = Xtest[used_features].drop(fi, axis=1)
        y_test_hat_i, f_used_i, rmse_train_i = model_LassoCV(X_,y,Xtest_, scaler=scaler, verbose=False)
        rmse_i = mean_squared_error(Ytest,y_test_hat_i)
        rmse_list.append([rmse_train_i, rmse_i, rmse_train_i-rmse_i,len(f_used_i)])
        print('\t', fi, rmse_train_i, rmse_i, rmse_train_i-rmse_i, len(f_used_i))
    rmses = pd.DataFrame(data=np.array(rmse_list), index=features_to_drop, columns=['rmse_train','rmse_test','diff','n_sed'])
    print(rmses['diff'].sort_values(ascending=False))
    
def xboost_model(X,y,Xtest, scaler=None, n_search=5):
    from xgboost import XGBRegressor
    param_dist = {
                    "learning_rate": rand_power(-2,0),
                    }
                    #~ "n_estimators": rand_uni_int(100,600),
                    #~ "max_depth": [2,3,4,5]
    #~ random_search = model_random_search(XGBRegressor(max_depth=2,
                                                     #~ n_estimators=400,
                                                     #~ learning_rate=0.05),
                                        #~ param_dist,
                                        #~ n_search, 
                                        #~ X, y)
    #~ model = random_search.best_estimator_
    
    model = XGBRegressor(colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2)
                         
                         
    model.fit(X, y, verbose=False)
    y_hat = model.predict(X)
    rmse = get_rmse(y,y_hat,scaler)
    print('* XGBoost model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest)
    
    
def tune_SVR(X,y,Xtest, n_search=5, scaler=None):
    from sklearn.svm import SVR
    param_dist_svr = {
                    "C": rand_uni(0,20),
                    "epsilon": rand_power(-3,0),
                    }
                    #~ "gamma": rand_power(-5,1)
    random_search_svr = model_random_search(SVR(),
                                            param_dist_svr,
                                            n_search, 
                                            X, y)
    model = random_search_svr.best_estimator_
    model.fit(X.values, y)
    y_hat = model.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    print('* NN Regressor model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest.values)
    
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
    
def tune_KernelRidge(X,y,Xtest, n_search=5, scaler=None):
    from sklearn.kernel_ridge import KernelRidge
    param_dist = {
                  "alpha": rand_power(-2,0),
                  "coef0": rand_uni(0,3)
                 }
                  #~ "gamma": rand_power(-4,-2)
    random_search = model_random_search(KernelRidge(kernel="polynomial", 
                                                    degree=4,
                                                    alpha=0.1,
                                                    gamma=0.001),
                                        param_dist,
                                        n_search, 
                                        X, y)
    model = random_search.best_estimator_
    model.fit(X.values, y)
    y_hat = model.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    print('* NN Regressor model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest.values)
    
def tune_RFR(X,y,Xtest, n_search=5, scaler=None):
    from sklearn.ensemble import RandomForestRegressor
    param_dist_rfr = {
                    "n_estimators": rand_uni_int(10,300)
                    }
                    #~ "gamma": rand_power(-5,1)
    random_search_rfr = model_random_search(RandomForestRegressor(),
                                            param_dist_rfr,
                                            n_search, 
                                            X, y)
    model = random_search_rfr.best_estimator_
    model.fit(X.values, y)
    y_hat = model.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    print('* NN Regressor model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest.values)
    
def rand_uni(c_min,c_max,n_max=1000):
    return list(np.random.uniform(c_min,c_max,n_max))
    
def rand_uni_int(c_min,c_max,n_max=1000):
    return list(np.random.randint(c_min,c_max,n_max))
    
    
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return 
            
def model_random_search(cl, param_dist, n_iter, X, y):
    random_search = RandomizedSearchCV(cl, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       random_state=666,
                                       cv=3,
                                       verbose=2,
                                       scoring="neg_mean_squared_error")
    random_search = random_search.fit(X, y)
    report(random_search.cv_results_, n_top=5)
    return random_search
    
def rand_power(p_min,p_max,n_max=1000):
    return list(10.**np.random.uniform(p_min,p_max,n_max))

# hidden layers: generates list of n_max tuples with 
# n_l_min--n_l_max integers, each between n_a_min and n_a_max
def rand_hidden_layer_sizes(n_l_min,n_l_max,n_a_min,n_a_max,n_max=1000):
    n_l = np.random.randint(n_l_min,n_l_max,n_max)
    list_hl = []
    for nl_i in n_l:
        list_hl.append(tuple(np.random.randint(n_a_min,n_a_max,nl_i)))
    return list_hl
    
def run_nn(X, y, Xtest, scaler=None, **param):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(**param)
    model.fit(X.values, y)
    y_hat = model.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    print('* NN Regressor model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest.values)

def tune_nn(X, y, Xtest, n_search=5, scaler=None, solver='lbfgs', fout=None):
    from sklearn.neural_network import MLPRegressor
    
    # parameters to be tuned by random CV search
    param_dist_nn = {
                    "alpha":rand_power(-2,1),
                    }
                    #~ "learning_rate_init":rand_power(-3,0),
                    #~ "hidden_layer_sizes": rand_hidden_layer_sizes(2,3,50,200)
                     
    random_search_nn1 = model_random_search(MLPRegressor(solver=solver, 
                                                         tol=1e-3, 
                                                         activation='tanh',
                                                         learning_rate_init=0.08,
                                                         learning_rate='adaptive',
                                                         alpha=1.5,
                                                         hidden_layer_sizes=(500,),
                                                         max_iter=1000),
                                            param_dist_nn, 
                                            n_search, 
                                            X, y)
    if fout is not None:
        f = open(fout,"wb")
        pickle.dump(random_search_nn1.cv_results_ ,f)
        f.close()
    model = random_search_nn1.best_estimator_
    model.fit(X.values, y)
    y_hat = model.predict(X.values)
    rmse = get_rmse(y,y_hat,scaler)
    print('* NN Regressor model:')
    print('\t RMSE on train set', rmse)
    return model.predict(Xtest.values)
        
def save_prediction(answer, file_out):
    np.savetxt(file_out, answer, header='Id,SalePrice', delimiter=',', fmt= '%i,%.1f', comments='')
    
def write_result(ytest_log, test_id, fout):
    answer = np.array([test_id, np.exp(ytest_log)]).T
    save_prediction(answer, fout)


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
    
    #######################################
    ### MISSING VALUES IN THE TRAIN SET ###
    
    print('* Missing values')
    
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
    
    # Apart from the above features that have missing values in the train set,
    # there are several features with 1--4 missing values in the test set.
    
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
    
    ### TotalBsmtSF
    #~ print(train.corr(method='spearman')[['TotalBsmtSF']])
    #~ print(pd.concat([train,test], axis=1)['TotalBsmtSF'].describe())
    #~ plot_scatter(train, '1stFlrSF', 'TotalBsmtSF')
    #~ pd.set_option("display.max_columns",101)
    #~ print( test[test['TotalBsmtSF'].isnull()] )
    
    # Spearman correlation coef with SalePrice ~0.6
    # also highly correlated with 1stFlrSF -- basement cannot be larger than 1st Floor
    # There is no basement for this house so fill with 0
    test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0.)
    
    ### BsmtUnfSF
    #~ print(train.corr(method='spearman')[['BsmtUnfSF']])
    #~ print(pd.concat([train,test], axis=1)['BsmtUnfSF'].describe())
    #~ pd.set_option("display.max_columns",101)
    #~ print( test[test['BsmtUnfSF'].isnull()] )
    
    # There is no basement for this house so fill with 0
    test['BsmtUnfSF'] = test['TotalBsmtSF'].fillna(0.)
    
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
    test = transform_qual_features(test, ['KitchenQual'])
    
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
    
    # convert quality features (that have not been converted yet) to numeric
    qual_features_to_num = ['ExterQual', 'ExterCond', 'HeatingQC', 'GarageQual', 'GarageCond', 'PoolQC']
    train = transform_qual_features(train, qual_features_to_num)
    test = transform_qual_features(test, qual_features_to_num)
    
    # dummy variables for categorical features 
    features_cat = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 
       'CentralAir', 'Electrical', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType','SaleCondition',
       'Neighborhood', 'Condition1', 'Condition2']
       #~ 'Heating',
    train = add_dummies(train, features_cat)
    test = add_dummies(test, features_cat)
    
    # Remove some dummy variables that are present in the train set but not in the test set
    # Otherwise results in an error later
    dummies_not_in_test_set = ['HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran', 'RoofMatl_Metal',
                               'RoofMatl_Roll', 'Electrical_Mix', 'MiscFeature_TenC', 'Condition2_RRAe',
                               'Condition2_RRAn', 'Condition2_RRNn']
    train = train.drop(dummies_not_in_test_set, axis=1)
    
    #~ print(train.columns)
    
    # check correlation with SalePrice and keep featured correlated more than given limit
    corr_lim_keep = 0.05
    
    # correlation matrix
    corr_train = train.corr(method='spearman')
    features_keep = corr_train[corr_train['SalePrice'].abs()>corr_lim_keep].index.tolist()
    
    # drop the target variable and those eliminated during missing values examinationa
    features_keep.remove('SalePrice')
    for fi in features_not_used_due_to_na:
        try: features_keep.remove(fi)
        except: pass
        
    print("\n* Features with correlation with SalePrice >", corr_lim_keep, '|', len(features_keep), 'features out of', train.shape[1])
    #~ print(features_keep)
    
    ################################################
    ### MUTUAL CORRELATIONS OF SELECTED FEATURES ###
    
    # remove features with mutual correlations higher than given limit
    
    ### ??? removing features based on mutual corr -- features might be correlated in an entangled way and 
    ### feature A that is decided to be kept due to corr with feature B, might be decided to remove due to
    ### correlation with feature C. This is more important for lower mutual_lim.
    
    # keeping the mutual correlation limit high will only remove very correlated features,
    # e.g. complementary dummy variables
    mutual_lim = 0.95
    features_to_drop_due_to_mutual_cor = mutual_corr_more_tha_limit(corr_train, features_keep, verbose=False)
    print('* Features to remove due to mutual correlations >', mutual_lim, '|', \
        len(features_to_drop_due_to_mutual_cor), 'features out of', len(features_keep))
    for fi in features_to_drop_due_to_mutual_cor:
        features_keep.remove(fi)
    print('* Features with highest correlation with SalePrice:')
    print(corr_train.loc['SalePrice',features_keep].abs().sort_values(ascending=False).head())
    
    features_num = train[features_keep].select_dtypes(exclude=['uint8']).columns.tolist()
    #~ plot_pairplot(train, features_num)
    #~ plot_correlation_heatmap(train, features=features_keep, fout=None, method='spearman', verbose=False)
    
    ################
    ### Outliers ###
    
    # Manually remove few outliers
    
    # Based on public kernels, it seems that removing outliers should help a bit.
    # Let's plot few numeric features that are highly correlated with SalePrice
    fe_plot = ['SalePrice', 'GrLivArea', 'YearBuilt', 'LotArea']
    plot_pairplot(train, features=fe_plot, fout='pariplot_most_correlated.png')
    
    # GrLivArea has two outliers -- large area but much cheaper than the general trand suggests
    # removing these helps to get better score
    print('* Removing outliers:')
    ind_remove = train[(train['GrLivArea']>4000.) & (train['SalePrice']<200000)].index
    print('\t GrLivArea', ind_remove)
    train = train.drop(ind_remove)
    
    # LotArea
    ind_remove = train[train['LotArea']>100000.].index
    print('\t LotArea', ind_remove)
    train = train.drop(ind_remove)
    
    # Year built
    # helps a little bit
    ind_remove = train[(train['YearBuilt']<1900.) & (train['SalePrice']>200000)].index
    print('\t YearBuilt', ind_remove)
    train = train.drop(ind_remove)
    
        
    plot_pairplot(train, features=fe_plot, fout='pariplot_most_correlated_noout.png')
    #################
    ### ML MODELS ###
    
    # copy data using only selected features
    X_train_all = train[features_keep].copy(deep=True)#.astype('float64')
    y_train_all = train['SalePrice'].copy(deep=True).astype('float64')
    X_test = test[features_keep].copy(deep=True)#.astype('float64')
    test_id = test['Id'].astype('int64')
        
    #~ missing_values_overview([X_train_all,X_test], include_corr=False)
    
    # transform skewed features -- substantially improves models
    X_train_all, X_test = transform_skewed_features(X_train_all, X_test)
    
    # scale to mean=0, std=1, and type float64
    X_train_all.loc[:,:], X_test.loc[:,:] = scale_data(X_train_all, X_test)
    # scale SalePrice -- remember the inverse transform and np.exp
    y_train_all, scaler_SalePrice = scale_and_transform_SalePrice(y_train_all)
    
    # From public kernels, I learned that this feature contributes to overfitting.
    # Not sure how to find that out. I tried dropping individual features one by one and 
    # comparing CV resutls, but it wasn't that indicative.
    X_train_all = X_train_all.drop('MSZoning_C (all)',axis=1)
    X_test = X_test.drop('MSZoning_C (all)',axis=1)
        
    ### NN
    # I tried to use NN with some hyperparameters tuning, adam and lbfgs solvers with regularization.
    # All models end up overfitting OR result in a score around 0.14.
    #~ y_test_log = tune_nn(X_train_all, y_train_all, X_test, n_search=5, scaler=scaler_SalePrice, solver='adam')
    
    # best NN so far, score~0.13576
    # corr_lim_keep=0.1
    #~ par = {'solver':'sgd', 
           #~ 'tol':1e-3, 
           #~ 'activation':'tanh',
           #~ 'hidden_layer_sizes':(100,111,50),
           #~ 'learning_rate':'adaptive',
           #~ 'max_iter':1000,
           #~ 'alpha': 1.6766539709815422, 
           #~ 'learning_rate_init': 0.014476599457340257
           #~ }
    #~ y_test_log = run_nn(X_train_all, y_train_all, X_test, scaler=scaler_SalePrice, **par)
    #~ write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'nn_sdg_1.csv')        
    
    # corr_keep = 0
    #~ par = {'solver':'adam', 
           #~ 'tol':1e-3, 
           #~ 'activation':'tanh',
           #~ 'hidden_layer_sizes':(500),
           #~ 'learning_rate':'adaptive',
           #~ 'max_iter':1000,
           #~ 'alpha': 0.25, 
           #~ 'learning_rate_init': 0.08
           #~ }
    #~ y_test_log = run_nn(X_train_all, y_train_all, X_test, scaler=scaler_SalePrice, **par)
    #~ write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'nn_adam.csv')        
    
    ### Linear regression model with Lasso regularization
    # Lasso with correlation limit to keep of 0.05 results in the best score so far (0.1252)
    y_test_log, used_features, rmse_model = model_LassoCV(X_train_all, y_train_all.values, X_test, scaler=scaler_SalePrice)
    write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'lasso_all_us_out_2.csv')
    
    #~ X_split_train = X_train_all.sample(frac=0.7,random_state=666)
    #~ y_split_train = y_train_all[X_split_train.index]
    #~ X_split_test = X_train_all.drop(X_split_train.index)
    #~ y_split_test = y_train_all.drop(X_split_train.index)
    
    #~ run_LassoCV_across_variables(X_split_train, y_split_train.values, X_split_test, y_split_test.values, features_to_drop=None, scaler=scaler_SalePrice)
    
    
    ### SVR
    #~ y_test_log = tune_SVR(X_train_all, y_train_all, X_test, n_search=10)
    #~ write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'svr_1.csv')
    
    ### Kernel Ridge
    #~ y_test_log = tune_KernelRidge(X_train_all, y_train_all, X_test, n_search=10)
    #~ write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'kr_1.csv')
    
    ### Random Forest regressor
    #~ y_test_log = tune_RFR(X_train_all, y_train_all, X_test, n_search=5)
    #~ write_result(scaler_SalePrice.inverse_transform(y_test_log), test_id, 'svr_1.csv')
    
    ### XBoost
    #~ y_test_log = xboost_model(X_train_all, y_train_all, X_test, n_search=10)
    
    ######################
    
