from tabnanny import verbose
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_sample_weight
from sklearn.utils.extmath import stable_cumsum
import numpy as np

def avg_auc(classifiers, data, verbose=0):
    features, labels = data
    avg_auc = 0
    scorings = {}
    for c in classifiers.keys():
        y_true = labels == c
        try:
            scoring = classifiers[c].decision_function(features)
        except AttributeError:
            scoring = classifiers[c].score_samples(features)
        auc = roc_auc_score(y_true, scoring)
        scorings[c] = auc
        if verbose:
            print('Class:', c, 'roc-auc: ', auc)
        avg_auc += auc
    avg_auc /= len(classifiers.keys())
    if verbose:
        print('Average ROC-AUC:', avg_auc)
    return avg_auc, scorings

def avg_cs(classifiers, data, cost_model, verbose=0):
    features, labels = data
    avg_cs = 0
    scorings = {}
    for c in classifiers.keys():
        y_true = labels == c
        try:
            scoring = classifiers[c].decision_function(features)
        except AttributeError:
            scoring = classifiers[c].score_samples(features)
        cs = custom_scoring(y_true, scoring, cost_model)
        scorings[c] = cs
        if verbose:
            print('Class:', c, 'custom score: ', cs)
        avg_cs += cs
    avg_cs /= len(classifiers.keys())
    if verbose:
        print('Average ROC-AUC:', avg_cs)
    return avg_cs, scorings

def custom_scoring(y_true, y_score, cost_model, sample_weight=None):
    if sample_weight is not None:
        sample_weight = compute_sample_weight(sample_weight, y_true)
    fp, tp, thresholds = binary_clf_curve(y_true, y_score, sample_weight=sample_weight)
    
    if len(fp) > 2:
        optimal_idxs = np.where(
                np.r_[True, np.logical_or(np.diff(fp, 2), np.diff(tp, 2)), True]
            )[0]
        fp = fp[optimal_idxs]
        tp = tp[optimal_idxs]
        thresholds = thresholds[optimal_idxs]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    fn = max(tp) - tp
    tn = max(fp) - fp

    scores = tn*cost_model['tn'] + fp*cost_model['fp'] + fn*cost_model['fn'] + tp*cost_model['tp']
    best_score = min(scores) 
    return best_score

def binary_clf_curve(y_true, y_score, sample_weight=None):
    y_true = y_true == 1.0
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]