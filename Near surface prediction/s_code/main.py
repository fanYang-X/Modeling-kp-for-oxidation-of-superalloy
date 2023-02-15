import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import split
import prediction

def wt_to_at(data):
    atom_features = ['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf']
    atom_mass = {'Ni': 58.69, 'Al': 26.98, 'Co': 58.93, 'Cr': 52, 'Mo': 95.95, 'Re': 186.2, 'Ru': 101.07,
                'Ti': 47.87, 'Ta': 180.94, 'W':  183.84, 'Hf': 178.49, 'Nb': 92.90, 'Si': 28.085, 'C': 12,
                'Y': 88.90, 'Ce': 140.12, 'B': 10.81}
    at_value = np.array(data.values) / np.array([atom_mass[j] for j in atom_features])
    sum_num = np.sum(at_value)
    at_value = at_value / sum_num
    for index, i in enumerate(['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf']):
        data[i] = at_value[:, index]
    return data

def get_lsw(data):
    e_features = ['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf']
    at_df = wt_to_at(data[e_features])
    Q_dict = {'Ni': 284, 'Al': 272.09, 'Co': 285.1, 'Cr': 170.7, 'Mo': 281.3, 
              'Re': 255, 'Ru': 304.4, 'Ti': 275, 'Ta': 251, 'W': 264, 'Hf': 250.1}
    D_dict = {'Ni':1.9*10**(-4), 'Al':1*10**(-3), 'Co':7.5*10**(-5), 'Cr':3*10**(-6),
              'Mo':1.15*10**(-4), 'Re':8.2*10**(-7), 'Ru':2.48*10**(-4),
              'Ti':4.1*10**(-4), 'Ta':2.19*10**(-5), 'W':8*10**(-6), 'Hf': 1.62*10**(-4)}
    D0 = np.zeros(len(data))
    Q0 = np.zeros(len(data))
    for i in e_features:
        D0 += at_df[i] / D_dict[i]
        Q0 += at_df[i] * 1000 * Q_dict[i]
    D = (1 / D0) * np.exp(-Q0 / 8.314 / 1373.15)
    data['D'] = D * 1000
    return data

s_path = 'C:/Users/lenovo/Desktop/NSModel/data/NS_data_s.csv'
s_data = pd.read_csv(s_path, encoding='gb2312')
s_data = s_data[s_data['alloy'] != '32Y3']
features = ['distance', 't', 'gamapie', 'Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf', 'Nb']
data, label = s_data[features].values, s_data['gamapie size'].values
cv_data, train_data, train_label, test_data, test_label = split.DataSplit(data, label, 2022).get_split()

params = {'RF': {'n_estimators': 1321, 'max_features': 0.983,
                 'max_depth': 3},
          'SVR': {'kernel': 'poly', 'C': 2.909}, 
          'KRR': {'kernel': 'rbf', 'alpha': 0.0622}, 
          'LGB': {'n_estimators': 2835, 'subsample': 0.9689, 
                  'colsample_bytree': 0.9453, 'reg_alpha':  2.0396, 'reg_lambda': 0.0283, 
                  'min_child_samples': 1},
          'ENET': {'alpha': 0.13646, 'l1_ratio': 0.0009114},
          'MLP': {'hidden_layer_sizes': (64, 64, 64, 64,)}}

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

## MLP model
mlp = MLPRegressor(**params['MLP'])
prediction.cv_prediction('MLP', mlp, cv_data, train_data, train_label, test_data, test_label, False)
