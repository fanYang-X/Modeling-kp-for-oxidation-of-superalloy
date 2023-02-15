import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import split
import prediction

n_path = 'C:/Users/lenovo/Desktop/NSModel/data/NS_data_n.csv'
n_data = pd.read_csv(n_path, encoding='gb2312')
n_data = n_data[n_data['alloy'] != '32Y3']
features = ['distance', 't', 'Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf', 'Nb']
data, label = n_data[features].values, n_data['gamapie shape factor'].values
cv_data, train_data, train_label, test_data, test_label = split.DataSplit(data, label, 2022).get_split()

params = {'RF': {'n_estimators': 300, 'max_features': 0.92,
                 'max_depth': 9},
          'SVR': {'kernel': 'rbf', 'C': 18.683}, 
          'KRR': {'kernel': 'rbf', 'alpha': 0.0114}, 
          'LGB': {'n_estimators': 3010, 'subsample': 0.9058, 
                  'colsample_bytree': 0.8263, 'reg_alpha': 0.0001883, 'reg_lambda': 9.9139, 
                  'min_child_samples': 2},
          'ENET': {'alpha': 0.152, 'l1_ratio': 0.000116}}

## RF model
rf = RandomForestRegressor(**params['RF'])
prediction.cv_prediction('RF', rf, cv_data, train_data, train_label, test_data, test_label, False)

## SVR model
svr = SVR(**params['SVR'])
prediction.cv_prediction('SVR', svr, cv_data, train_data, train_label, test_data, test_label, False)

## KRR model
krr = KernelRidge(**params['KRR'])
prediction.cv_prediction('KRR', krr, cv_data, train_data, train_label, test_data, test_label, False)

## LGB model
LGB = lgb.LGBMRegressor(**params['LGB'], silent=False, verbosity=-1)
prediction.cv_prediction('LGB', LGB, cv_data, train_data, train_label, test_data, test_label, False)

## ENET model
enet = ElasticNet(**params['ENET'])
prediction.cv_prediction('ENET', enet, cv_data, train_data, train_label, test_data, test_label, False)
