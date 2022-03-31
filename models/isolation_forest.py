import sys
import tuning
from sklearn.ensemble import IsolationForest

def tune_isolation_forest(classes, data_train, data_val, estimator=None, params=None, verbose=0, scoring='roc_auc'):
    if params is None:
        params = {'n_estimators': [100],
                  'max_samples': ['auto', 0.1, 0.2, 0.5, 1.0]}
    if estimator is None:
        estimator = IsolationForest(random_state=42, contamination=1e-16)
    best_params = tuning.multiclass_tuning(estimator, params, classes, data_train, data_val, verbose=verbose, scoring=scoring)
    return best_params

def train_isolation_forest(classes, data_train, best_params):
    classifiers = {}
    features_train, labels_train = data_train
    for c in classes:
        params = best_params[c]
        classifier = IsolationForest(random_state=42, contamination=1e-16, n_estimators=params['n_estimators'], max_samples=params['max_samples'])
        classifier.fit(features_train[labels_train == c])
        classifiers[c] = classifier 
    return classifiers
