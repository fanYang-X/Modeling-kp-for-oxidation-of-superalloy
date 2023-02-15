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
import warnings
warnings.filterwarnings('ignore')

def cv(train_data, train_label):
    cv_data = {}
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    for fold, (train_id, val_id) in enumerate(kfold.split(train_data, train_label)):
        cv_data[fold] = {'train': (train_data[train_id], train_label[train_id]),
                         'val': (train_data[val_id], train_label[val_id])}
    return cv_data

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
    train['gamapie_pre'] = np.mean(train_pre, axis=1)
    init_test['gamapie_pre'] = np.mean(test_pre, axis=1)
    train['std'] = np.std(train_pre, axis=1)
    init_test['std'] = np.std(test_pre, axis=1)

    plt.figure(figsize=(15, 7.5))
    for index, i in enumerate([3, 6, 10, 20, 50, 100]):
        plt.subplot(2, 3, index + 1)
        plt.grid(linestyle='--', alpha=0.5, zorder=0)
        # plt.scatter(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['size_pre'], s=50, label='Predicted -{}h ML'.format(i), c='green', alpha=0.5)
        plt.scatter(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie fraction'], marker='^', s=50, label='Experimental -{}h'.format(i), c='red')
        plt.plot(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie fraction'], '--', c='red', alpha=0.8)
        plt.plot(init_test[init_test['t'] == i]['distance'], 
                 init_test[init_test['t'] == i]['gamapie_pre'], '--', label='{} (TC Com.) Predicted -{}h'.format(model_name, i), linewidth=2.5, c='royalblue')
        plt.fill_between(init_test[init_test['t'] == i]['distance'], init_test[init_test['t'] == i]['gamapie_pre'] + init_test[init_test['t'] == i]['std'], 
                         init_test[init_test['t'] == i]['gamapie_pre'] - init_test[init_test['t'] == i]['std'], facecolor='royalblue', alpha=0.3)
        plt.errorbar(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['gamapie_pre'],
                     train[train['t'] == i]['std'], fmt='o', mfc='tomato', mec='tomato', ecolor='gray',
                     ms=7, elinewidth=0.4, capsize=2, label='{} (Measured Com.) Predicted -{}h'.format(model_name, i), capthick=0.4, alpha=0.7)
        plt.plot(train[(train['t'] == i)]['distance'] + val_data[val_data['t'] == i]['distance'].values[1], train[train['t'] == i]['gamapie_pre'], '--', c='tomato', alpha=0.5)
        plt.plot(init_test[init_test['t'] == i]['distance'], 
                 init_test[init_test['t'] == i]['Al'] / 10, '--', label='{} Al -{}h'.format('TC', i), linewidth=2.5, c='green')
        plt.xlabel('Distance (um)', {'family': 'Times New Roman', 'weight': 'bold', 'size': 9})
        plt.ylabel("γ' fraction", {'family': 'Times New Roman', 'weight': 'bold', 'size': 9})
        plt.xlim(0, val_data[val_data['t'] == i]['distance'].max() + 2)
        # plt.ylim(0, val_data[val_data['t'] == i]['gamapie size'].max() + 1)
        plt.ylim(0, 1)
        plt.legend(loc=2, frameon=False)
        '''plt.subplot(2, 3, index + 1)
        plt.scatter(val_data[(val_data['t'] == i)&(val_data['gamapie fraction'] > 0)]['distance'], train[train['t'] == i]['gamapie_pre'], s=50, label='Predicted -{}h ML'.format(i), c='green', alpha=0.5)
        plt.plot(val_data[(val_data['t'] == i)&(val_data['gamapie fraction'] > 0)]['distance'], train[train['t'] == i]['gamapie_pre'], '--', c='green', alpha=0.5)
        plt.scatter(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie fraction'], marker='^', s=50, label='Experimental -{}h'.format(i), c='red')
        plt.plot(val_data[val_data['t'] == i]['distance'], val_data[val_data['t'] == i]['gamapie fraction'], '--', c='red', alpha=0.8)
        plt.plot(init_test[init_test['t'] == i]['distance']*(10**6), 
                 init_test[init_test['t'] == i]['gamapie_pre'], '--', label='{} Predicted -{}h Base TC'.format(model_name, i), linewidth=2.5, c='royalblue')
        plt.xlabel('Distance (um)', font)
        plt.ylabel("γ' fraction", font)
        plt.xlim(0, val_data[val_data['t'] == i]['distance'].max() + 2)
        # plt.ylim(0, 1)
        plt.legend(loc=4)'''
        if save:
            plt.savefig('C:/Users/lenovo/Desktop/NSModel/figure/f/{}_f_ML_predicted.tif'.format(model_name), dpi=330, bbox_inches='tight')
    plt.show()
    train.drop(['gamapie_pre'], axis=1, inplace=True)

f_path = 'C:/Users/lenovo/Desktop/NSModel/data/NS_data_f.csv'
f_data = pd.read_csv(f_path, encoding='gb2312')
f_data0 = f_data[f_data['alloy'] == '32Y3']
f_data = f_data[f_data['alloy'] != '32Y3']
features = ['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf', 'Nb']
data, label = f_data[features].values, f_data['gamapie fraction'].values
cv_data = cv(data, label)

# 32Y3 data
y3_tc_data = pd.read_csv('D:/ML kp/TC_clac/1100_elemental.csv', encoding='gb2312')
y3_data = pd.read_csv('C:/Users/lenovo/Desktop/y3_f.csv', encoding='gb2312')
for i in ['Ni', 'Al', 'Ta', 'Mo', 'Re']:
    y3_tc_data[i] = y3_tc_data[i] * 100
y3_tc_data['distance'] = y3_tc_data['distance'] * (10**6)
test_data = y3_tc_data[features].values

params = {'RF': {'n_estimators': 104, 'max_features': 1,
                 'max_depth': 6, 'random_state': 2022},
          'SVR': {'kernel': 'rbf', 'C': 4.489}, 
          'KRR': {'kernel': 'poly', 'alpha':  0.0936}, 
          'LGB': {'n_estimators': 2785, 'subsample': 0.8, 
                  'colsample_bytree': 1, 'reg_alpha': 0.1105, 'reg_lambda': 0.08555, 
                  'min_child_samples': 2},
          'ENET': {'alpha': 0.001526, 'l1_ratio': 0.0021274}}

## RF model
rf = RandomForestRegressor(**params['RF'])
cv_prediction('RF', rf, cv_data, f_data0, test_data, y3_tc_data, y3_data, True)

## SVR model
svr = SVR(**params['SVR'])
cv_prediction('SVR', svr, cv_data, f_data0, test_data, y3_tc_data, y3_data, True)

## KRR model
krr = KernelRidge(**params['KRR'])
cv_prediction('KRR', krr, cv_data, f_data0, test_data, y3_tc_data, y3_data, True)

## LGB model
LGB = lgb.LGBMRegressor(**params['LGB'], silent=False, verbosity=-1)
cv_prediction('LGB', LGB, cv_data, f_data0, test_data, y3_tc_data, y3_data, True)

## ENET model
enet = ElasticNet(**params['ENET'])
cv_prediction('ENET', enet, cv_data, f_data0, test_data, y3_tc_data, y3_data, True)
