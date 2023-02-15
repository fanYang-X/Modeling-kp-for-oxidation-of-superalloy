import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
plt.rc('font', family='Times New Roman', size=7.5, weight='bold')
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 7.5}

def cv_prediction(model_name, model, cv_data, train, train_label, test, test_label, save=False):
    train_pre = np.zeros(len(train))
    test_pre = np.zeros(len(test))
    for fold in range(len(cv_data)):
        trn_data, trn_label = cv_data[fold]['train'][0], cv_data[fold]['train'][1]
        scalar = StandardScaler()
        trn_data = scalar.fit_transform(trn_data)
        train_data = scalar.transform(train)
        test_data = scalar.transform(test)
        model.fit(trn_data, trn_label)
        train_pre += model.predict(train_data) / len(cv_data)
        test_pre += model.predict(test_data) / len(cv_data)
    
    plt.figure(figsize=(3.5, 3.5))
    plt.scatter(train_pre, train_label, c='royalblue', marker='o', label='Train', edgecolors='blue')
    plt.scatter(test_pre, test_label, c='tomato', marker='o', label='Test', edgecolors='red')
    plt.plot(range(-100, 100), range(-100, 100), c='gray')
    plt.text(0.05, 0.75, s='RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(test_label, test_pre))))
    plt.text(0.05, 0.7, s='MAE: {:.3f}'.format(mean_absolute_error(test_label, test_pre)))
    plt.text(0.05, 0.65, s= '$\mathregular{R^2}$' + ': {:.3f}'.format(r2_score(test_label, test_pre)))
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1)
    plt.xlabel('Predicted', font)
    plt.ylabel('Experimental', font)
    plt.legend()
    if save:
        plt.savefig('C:/Users/chonglh/Desktop/oxidaton2.0/nearsurface/NS_ml/code/n/result/{}_n_test.tif'.format(model_name), dpi=330, bbox_inches='tight')
    plt.show()

    print('{} Performance on test'.format(model_name))
    print('MAE: {:.4f}'.format(mean_absolute_error(test_label, test_pre)))
    print('RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(test_label, test_pre))))
    print('R2: {:.4f}'.format(r2_score(test_label, test_pre)))