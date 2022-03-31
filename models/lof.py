import tuning
from sklearn.neighbors import LocalOutlierFactor


def tune_lof(classes, data_train, data_val, estimator=None, params=None, verbose=0, scoring='roc_auc'):
    if params is None:
        params = {'n_neighbors': list(range(100, 1001, 100))}
    if estimator is None:
        estimator = LocalOutlierFactor(novelty=True, contamination='auto')
    best_params = tuning.multiclass_tuning(estimator, params, classes, data_train, data_val, verbose=verbose, scoring=scoring)
    return best_params

def train_lof(classes, data_train, best_params):
    classifiers = {}
    features_train, labels_train = data_train
    for c in classes:
        params = best_params[c]
        classifier = LocalOutlierFactor(novelty=True, contamination='auto', n_neighbors=params['n_neighbors'])
        classifier.fit(features_train[labels_train == c])
        classifiers[c] = classifier 
    return classifiers