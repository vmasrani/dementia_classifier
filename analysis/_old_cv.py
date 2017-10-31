# import pandas as pd
import numpy as np
import data_handler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from dementia_classifier.analysis import util
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from sklearn.linear_model import LogisticRegression
from dementia_classifier.analysis.cross_validators import DementiaCV
# models
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression

# globals
XTICKS = np.arange(1, 300, 1)
REGULARIZATION_CONSTANT = 1


# def evaluate_model(model, X, y, labels, save_features=False, group_ablation=False, feature_output_name="features.csv", ticks=XTICKS):

#     model_fs = RandomizedLogisticRegression(C=1, random_state=1)
#     # Split into folds using labels

#     group_kfold = GroupKFold(n_splits=10).split(X, y, groups=labels)
#     folds_fmeas  = []

#     for train_index, test_index in group_kfold:
#         print "processing fold: %d" % (len(folds_fmeas) + 1)

#         # Split
#         X_train, X_test = X.values[train_index], X.values[test_index]
#         y_train, y_test = y.values[train_index], y.values[test_index]

#         scores_fmeas = []

#         nfeats = X_train.shape[1]
#         feats = util.get_top_pearson_features(X_train, y_train, nfeats)

#         for k in XTICKS:
#             indices = feats[:k]
            
#             # Select k features
#             X_train_fs = X_train[:, indices]
#             X_test_fs  = X_test[:, indices]

#             model = model.fit(X_train_fs, y_train)
#             # summarize the selection of the attributes
#             yhat  = model.predict(X_test_fs)                  # Predict
#             scores_fmeas.append(f1_score(y_test, yhat))       # Save
            
#         # ----- save row -----
#         folds_fmeas.append(scores_fmeas)
#         # ----- save row -----

#     folds_fmeas = np.asarray(folds_fmeas)
#     return folds_fmeas


# def run_cv():
#     model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
#     new_features = feature_set_list.get_new_features()
#     X, y, labels = data_handler.get_data(drop_features=new_features, polynomial_terms=None)
#     folds_fmeas = evaluate_model(model, X, y, labels, f1_score)
#     print "f1_score"
#     print folds_fmeas.mean(axis=0).max()
    
    # print "accuracy_score"
    # print folds_fmeas.mean(axis=0).max()


def run_cv():
    model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
    new_features = feature_set_list.get_new_features()
    X, y, labels = data_handler.get_data(drop_features=new_features, polynomial_terms=None)
    CV = DementiaCV(X, y, labels, model, f1_score).train_model(k_range=XTICKS)
    print CV.best_score