import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import split

f_path = 'D:/ML kp/NS_ml_revised/data/NS_data_f.csv'
f_data = pd.read_csv(f_path, encoding='gb2312')
f_data = f_data[f_data['alloy'] != '32Y3']
data, label = f_data.values[:, 15: -1], f_data['gamapie fraction'].values
cv_data, _, _, test_data, test_label = split.DataSplit(data, label, 2022).get_split()
params = {}

'''## RandomForest
def objective(trial):
    mae_list = []
    param = {'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
             'max_features': trial.suggest_uniform('max_features', 0.7, 1), 
             'max_depth': trial.suggest_int("max_depth", 2, 12, log=True)}
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        val_data, val_label = cv_data[fold]['val'][0], cv_data[fold]['val'][1]
        clf = RandomForestRegressor(**param)
        clf.fit(trn_data, trn_label)
        pre_val = clf.predict(val_data)
        mae_list.append(mean_absolute_error(val_label, pre_val))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
  
trial = study.best_trial
params['RF'] = (trial.params.items(), trial.value)'''

## SVR
def objective(trial):
    mae_list = []
    param = {'kernel': trial.suggest_categorical('kernel',['linear', 'poly', 'rbf', 'sigmoid']), 
             'C': trial.suggest_float("C", 1e-5, 1e3, log=True)}
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        val_data, val_label = cv_data[fold]['val'][0], cv_data[fold]['val'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        val_data = scalar.transform(val_data)
        clf = SVR(**param)
        clf.fit(trn_data, trn_label)
        pre_val = clf.predict(val_data)
        mae_list.append(mean_absolute_error(val_label, pre_val))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

trial = study.best_trial
params['SVR'] = (trial.params.items(), trial.value)

## KRR
def objective(trial):
    mae_list = []
    param = {'kernel': trial.suggest_categorical('kernel',['linear', 'poly', 'rbf', 'sigmoid']), 
             'alpha': trial.suggest_float("alpha", 1e-5, 1e3, log=True)}
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        val_data, val_label = cv_data[fold]['val'][0], cv_data[fold]['val'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        val_data = scalar.transform(val_data)
        clf = KernelRidge(**param)
        clf.fit(trn_data, trn_label)
        pre_val = clf.predict(val_data)
        mae_list.append(mean_absolute_error(val_label, pre_val))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

trial = study.best_trial
params['KRR'] = (trial.params.items(), trial.value)

'''## LGB
def objective(trial):
    mae_list = []
    param = {'n_estimators': trial.suggest_int('n_estimators', 2000, 5000), 
             'subsample': trial.suggest_uniform('subsample', 0.5, 1),
             'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
             'reg_alpha': trial.suggest_float("reg_alpha", 0.00001, 10, log=True),
             'reg_lambda': trial.suggest_float("reg_lambda", 0.00001, 10, log=True), 
             'min_child_samples': trial.suggest_int('min_child_samples', 1, 2)}
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        val_data, val_label = cv_data[fold]['val'][0], cv_data[fold]['val'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        val_data = scalar.transform(val_data)
        clf = lgb.LGBMRegressor(**param, silent=False, verbosity=-1)
        clf.fit(trn_data, trn_label)
        pre_val = clf.predict(val_data)
        mae_list.append(mean_absolute_error(val_label, pre_val))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

trial = study.best_trial
params['LGB'] = (trial.params.items(), trial.value)'''

## ENet
def objective(trial):
    mae_list = []
    param = {'alpha': trial.suggest_float("alpha", 1e-5, 1, log=True), 
             'l1_ratio': trial.suggest_float("l1_ratio", 1e-5, 1, log=True)}
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        val_data, val_label = cv_data[fold]['val'][0], cv_data[fold]['val'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        val_data = scalar.transform(val_data)
        clf = ElasticNet(**param)
        clf.fit(trn_data, trn_label)
        pre_val = clf.predict(val_data)
        mae_list.append(mean_absolute_error(val_label, pre_val))
    return np.mean(mae_list)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

trial = study.best_trial
params['ENet'] = (trial.params.items(), trial.value)