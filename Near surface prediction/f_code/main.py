import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import split
import prediction
import warnings
warnings.filterwarnings('ignore')

f_path = 'D:/ML kp/NS_ml_revised/data/NS_data_f.csv'
f_data = pd.read_csv(f_path, encoding='gb2312')
f_data = f_data[f_data['alloy'] != '32Y3']
data, label = f_data.values[:, 15: -1], f_data['gamapie fraction'].values
cv_data, train_data, train_label, test_data, test_label = split.DataSplit(data, label, 2022).get_split()

params = {'RF': {'n_estimators': 104, 'max_features': 0.94242,
                 'max_depth': 6, 'random_state': 2022},
          'SVR': {'kernel': 'rbf', 'C': 4.48598}, 
          'KRR': {'kernel': 'poly', 'alpha': 0.0936}, 
          'LGB': {'n_estimators': 2785, 'subsample': 0.8, 
                  'colsample_bytree': 0.6943, 'reg_alpha': 0.1105, 'reg_lambda': 0.08555, 
                  'min_child_samples': 2},
          'ENET': {'alpha': 0.00037178, 'l1_ratio': 0.0006409}}

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
