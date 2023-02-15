import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
plt.rc('font', family='Times New Roman', size=7.5, weight='bold')
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 7.5}
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

def cv(train_data, train_label):
    cv_data = {}
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    for fold, (train_id, val_id) in enumerate(kfold.split(train_data, train_label)):
        cv_data[fold] = {'train': (train_data[train_id], train_label[train_id]),
                         'val': (train_data[val_id], train_label[val_id])}
    return cv_data

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

def cv_prediction(model_name, model, cv_data, train, test, init_test, val_data, save=False):
    train_pre = np.zeros((len(train), len(cv_data)))
    test_pre = np.zeros((len(test), len(cv_data)))
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        train_data = scalar.transform(train[features])
        test_data = scalar.transform(test)
        model.fit(trn_data, trn_label)
        train_pre[:, fold] = model.predict(train_data)
        test_pre[:, fold] = model.predict(test_data)
    train['size_pre'] = np.mean(train_pre, axis=1)
    init_test['size_pre'] = np.mean(test_pre, axis=1)
    train['std'] = np.std(train_pre, axis=1)
    init_test['std'] = np.std(test_pre, axis=1)

    plt.figure(figsize=(15, 7.5))
    for index, i in enumerate([3, 6, 10, 20, 50, 100]):
        plt.subplot(2, 3, index + 1)
        plt.grid(linestyle='--', alpha=0.5, zorder=0)
        # plt.scatter(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['size_pre'], s=50, label='Predicted -{}h ML'.format(i), c='green', alpha=0.5)
        plt.scatter(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie size'], marker='^', s=50, label='Experimental -{}h'.format(i), c='red')
        plt.plot(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie size'], '--', c='red', alpha=0.8)
        plt.plot(init_test[init_test['t'] == i]['distance'], 
                 init_test[init_test['t'] == i]['size_pre'], '--', label='{} (TC Com.) Predicted -{}h'.format(model_name, i), linewidth=2.5, c='royalblue')
        plt.fill_between(init_test[init_test['t'] == i]['distance'], init_test[init_test['t'] == i]['size_pre'] + init_test[init_test['t'] == i]['std'], 
                         init_test[init_test['t'] == i]['size_pre'] - init_test[init_test['t'] == i]['std'], facecolor='royalblue', alpha=0.3)
        plt.errorbar(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['size_pre'],
                     train[train['t'] == i]['std'], fmt='o', mfc='tomato', mec='tomato', ecolor='tomato',
                     ms=7, elinewidth=0.4, capsize=2, label='{} (Measured Com.) Predicted -{}h'.format(model_name, i), capthick=0.4, alpha=0.7)
        plt.plot(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['size_pre'], '--', c='tomato', alpha=0.5)
        plt.plot(init_test[init_test['t'] == i]['distance'], 
                 init_test[init_test['t'] == i]['Al'] / 10, '--', label='{} Al -{}h'.format('TC', i), linewidth=2.5, c='green')
        plt.xlabel('Distance (um)', {'family': 'Times New Roman', 'weight': 'bold', 'size': 9})
        plt.ylabel("γ' size (um*um)", {'family': 'Times New Roman', 'weight': 'bold', 'size': 9})
        plt.xlim(0, val_data[val_data['t'] == i]['distance'].max() + 2)
        # plt.ylim(0, val_data[val_data['t'] == i]['gamapie size'].max() + 1)
        plt.ylim(0, 5)
        plt.legend(loc=2, frameon=False)
        if save:
            plt.savefig('C:/Users/lenovo/Desktop/NSModel/figure/s/{}_s_ML_predicted.tif'.format(model_name), dpi=330, bbox_inches='tight')
    plt.show()

s_path = 'C:/Users/lenovo/Desktop/NSModel/data/NS_data_s.csv'
s_data = pd.read_csv(s_path, encoding='gb2312')
s_data['t_'] = s_data['t']**(1/3)
s_data0 = s_data[s_data['alloy'] == '32Y3']
s_data = s_data[s_data['alloy'] != '32Y3']
# s_data = get_lsw(s_data)
features = ['distance', 't_', 'gamapie', 'Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf', 'Nb']
data, label = s_data[features].values, s_data['gamapie size'].values
cv_data = cv(data, label)

# 32Y3 data
y3_tc_data = pd.read_csv('D:/ML kp/TC_clac/1100_elemental.csv', encoding='gb2312')
y3_data = pd.read_csv('D:/ML kp/NS_ml_revised/data/y3_s.csv', encoding='gb2312')
for i in ['Ni', 'Al', 'Ta', 'Mo', 'Re']:
    y3_tc_data[i] = y3_tc_data[i] * 100
y3_tc_data['distance'] = y3_tc_data['distance'] * (10**6)
y3_tc_data['t_'] = y3_tc_data['t']**(1/3)
# y3_tc_data = get_lsw(y3_tc_data)
test_data = y3_tc_data[features].values

params = {'RF': {'n_estimators': 100, 'max_features': 0.983,
                 'max_depth': 3},
          'SVR': {'kernel': 'poly', 'C': 20}, 
          'KRR': {'kernel': 'rbf', 'alpha': 0.1622}, 
          'LGB': {'n_estimators': 2835, 'subsample': 0.9689, 
                  'colsample_bytree': 0.9453, 'reg_alpha':  2.0396, 'reg_lambda': 0.0283, 
                  'min_child_samples': 1},
          'ENET': {'alpha': 0.13646, 'l1_ratio': 0.0009114},
          'MLP': {'hidden_layer_sizes': (64, 64, 64, 64,)}}

## RF model
rf = RandomForestRegressor(**params['RF'])
cv_prediction('RF', rf, cv_data, s_data0, test_data, y3_tc_data, y3_data, True)

## SVR model
svr = SVR(**params['SVR'])
cv_prediction('SVR', svr, cv_data, s_data0, test_data, y3_tc_data, y3_data, True)

## KRR model
krr = KernelRidge(**params['KRR'])
cv_prediction('KRR', krr, cv_data, s_data0, test_data, y3_tc_data, y3_data, True)

## LGB model
LGB = lgb.LGBMRegressor(**params['LGB'], silent=False, verbosity=-1)
cv_prediction('LGB', LGB, cv_data, s_data0, test_data, y3_tc_data, y3_data, True)

## ENET model
enet = ElasticNet(**params['ENET'])
cv_prediction('ENET', enet, cv_data, s_data0, test_data, y3_tc_data, y3_data, True)

## MLP model
# mlp = MLPRegressor(**params['MLP'])
# cv_prediction('MLP', mlp, cv_data, s_data0, test_data, y3_tc_data, y3_data, False)
