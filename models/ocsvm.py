import numpy as np

from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import tuning

def tune_ocsvm(classes, data_train, data_val, estimator=None, params=None, verbose=0, scoring='roc_auc'):
    if params is None:
        params = {'nu': [0.0001, 0.001, 0.01, 0.1, 0.5],
                  'gamma': np.linspace(5.5, 6.5, 11)}
    if estimator is None:
        estimator = OneClassSVM()
    best_params = tuning.multiclass_tuning(estimator, params, classes, data_train, data_val, verbose=verbose, scoring=scoring)
    return best_params

def tune_sgd_ocsvm(classes, data_train, data_val, estimator=None, params=None, verbose=0, scoring='roc_auc'):
    if params is None:
        params = {'sgdoneclasssvm__nu': [0.0001, 0.001, 0.01, 0.1, 0.5],
                  'nystroem__gamma': np.linspace(5.5, 6.5, 11)}
    if estimator is None:
        kernel_approx = Nystroem(random_state=42)
        ocsvm = SGDOneClassSVM(random_state=42, max_iter=1000)
        estimator = make_pipeline(kernel_approx, ocsvm)
    best_params = tuning.multiclass_tuning(estimator, params, classes, data_train, data_val, verbose=verbose, scoring=scoring)
    return best_params

def train_ocsvm(classes, data_train, best_params):
    classifiers = {}
    features_train, labels_train = data_train
    for c in classes:
        params = best_params[c]
        classifier = OneClassSVM(nu=params['nu'], gamma=params['gamma'])
        classifier.fit(features_train[labels_train == c])
        classifiers[c] = classifier 
    return classifiers

def train_sgd_ocsvm(classes, data_train, best_params):
    classifiers = {}
    features_train, labels_train = data_train
    for c in classes:
        params = best_params[c]
        kernel_approx = Nystroem(random_state=42, gamma=params['nystroem__gamma'])
        ocsvm = SGDOneClassSVM(random_state=42, max_iter=1000, nu=params['sgdoneclasssvm__nu'])
        classifier = make_pipeline(kernel_approx, ocsvm)
        classifier.fit(features_train[labels_train == c])
        classifiers[c] = classifier 
    return classifiers
