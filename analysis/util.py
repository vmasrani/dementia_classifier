import pandas as pd
import numpy as np
from pandas.io import sql
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.analysis import data_handler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

REGULARIZATION_CONSTANT = 1


CLASSIFIERS = {
    "DummyClassifier": DummyClassifier("most_frequent"),
    "LogReg": LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT),
    "KNeighbors": KNeighborsClassifier(3),
    "RandomForest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "MLP": MLPClassifier(alpha=1),
    "GausNaiveBayes": GaussianNB(),
}

CLASSIFIERS_KEYS = [
    "DummyClassifier",
    "LogReg",
    "KNeighbors",
    "RandomForest",
    "MLP",
    "GausNaiveBayes",
]

ABLATION_SETS = [
    "none",
    "cfg",
    "syntactic_complexity",
    "psycholinguistic",
    "vocabulary_richness",
    "repetitiveness",
    "acoustics",
    "parts_of_speech",
    "information_content",
    "demographic",
]


BLOG_ABLATION_SETS = [
    "none",
    "cfg",
    "syntactic_complexity",
    "psycholinguistic",
    "vocabulary_richness",
    "repetitiveness",
    "parts_of_speech",
]


NEW_FEATURE_SETS = [
    "none",
    "strips",
    "halves",
    "quadrant",
    "discourse",
]


def ablation_dataset_helper(key):
    to_drop = feature_set_list.new_features()
    if key == "cfg":
        to_drop += feature_set_list.cfg_features()
    elif key == "syntactic_complexity":
        to_drop += feature_set_list.syntactic_complexity_features()
    elif key == "psycholinguistic":
        to_drop += feature_set_list.psycholinguistic_features()
    elif key == "vocabulary_richness":
        to_drop += feature_set_list.vocabulary_richness_features()
    elif key == "repetitiveness":
        to_drop += feature_set_list.repetitiveness_features()
    elif key == "acoustics":
        to_drop += feature_set_list.acoustics_features()
    elif key == "parts_of_speech":
        to_drop += feature_set_list.parts_of_speech_features()
    elif key == "information_content":
        to_drop += feature_set_list.information_content_features()
    elif key == "demographic":
        to_drop += feature_set_list.demographic_features()
    elif key == "none":
        pass
    else:
        raise ValueError("Incorrect key")
    
    return data_handler.get_data(drop_features=to_drop)


def blog_feature_ablation(key):
    to_drop = []
    if key == "cfg":
        to_drop += feature_set_list.cfg_features()
    elif key == "syntactic_complexity":
        to_drop += feature_set_list.syntactic_complexity_features()
        to_drop.remove('number_of_utterances')
        to_drop.append('number_of_sentences')
    elif key == "psycholinguistic":
        to_drop += feature_set_list.psycholinguistic_features()
    elif key == "vocabulary_richness":
        to_drop += feature_set_list.vocabulary_richness_features()
    elif key == "repetitiveness":
        to_drop += feature_set_list.repetitiveness_features()
    elif key == "parts_of_speech":
        to_drop += feature_set_list.parts_of_speech_features()
    elif key == "none":
        pass
    else:
        raise ValueError("Incorrect key")    
    return data_handler.get_blog_data(drop_features=to_drop)


def new_features_dataset_helper(key):
    to_drop = feature_set_list.new_features()
    if key == "strips":
        new_feature_set = feature_set_list.strips_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "halves":
        new_feature_set = feature_set_list.halves_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "quadrant":
        new_feature_set = feature_set_list.quadrant_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
    elif key == "discourse":
        new_feature_set = feature_set_list.discourse_features()
        to_drop = [f for f in to_drop if f not in new_feature_set]
        new_feature_set = []
    elif key == "none":
        return data_handler.get_data(drop_features=to_drop)
    else:
        raise ValueError("Incorrect key")
        
    return data_handler.get_data(drop_features=to_drop, polynomial_terms=new_feature_set)


def delete_sql_tables(bad_tables):
    for table in bad_tables:
        print 'Deleting %s' % table
        sql.execute('DROP TABLE IF EXISTS %s' % table, cnx)


def get_top_pearson_features(X, y, n, return_correlation=False):
    df = pd.DataFrame(X).apply(pd.to_numeric)
    df['y'] = y
    corr_coeff = df.corr()['y'].abs().sort_values(inplace=False, ascending=False)
    if return_correlation:
        return corr_coeff
    else:
        return corr_coeff.index.values[1:n + 1].astype(int)


def whiten_matrix(X):
    '''Computes the square root matrix of symmetric square matrix X.'''
    (L, V) = np.linalg.eigh(X)
    return V.dot(np.diag(np.power(L, -0.5))).dot(V.T)


def color_matrix(X):
    '''Computes the square root matrix of symmetric square matrix X.'''
    (L, V) = np.linalg.eigh(X)
    return V.dot(np.diag(np.power(L, 0.5))).dot(V.T)


def trim(X, y, percentage):
    if percentage < 1:
        init = float(X.shape[0])
        for l in set(X.index):
            X = X.drop(l)
            y = y.drop(l)
            if X.shape[0] / init < percentage:
                break
    return X, y
