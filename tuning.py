import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.utils import compute_sample_weight
from sklearn.metrics import make_scorer


def hyperparameter_tuning(estimator, params, normal_data_train, x_val, y_val, verbose=0, scoring='roc_auc'):
    features = np.vstack([normal_data_train, x_val])
    val_fold = np.concatenate([[-1]*normal_data_train.shape[0], [0]*len(x_val)])
    y_true = np.concatenate([[1]*normal_data_train.shape[0], y_val])

    ps = PredefinedSplit(test_fold=val_fold)
    model = GridSearchCV(estimator, params, scoring=scoring, refit=False, cv=ps, verbose=verbose, n_jobs=-1)
    model.fit(features, y_true)
    return model.best_params_

def multiclass_tuning(estimator, params, classes, data_train, data_val, verbose=0, scoring='roc_auc'):
    best_params = {}
    for c in classes:
        y_val_c = data_val[1] == c
        normal_data_train = data_train[0][data_train[1] == c]
        best_params[c] = hyperparameter_tuning(estimator, params, normal_data_train, data_val[0], y_val_c, verbose=verbose, scoring=scoring)
        print('Class id ',c ,': best parameters: ', best_params[c], sep='')
    return best_params