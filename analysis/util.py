import pandas as pd
import numpy as np
from pandas.io import sql
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.analysis import data_handler
import models

from dementia_classifier.settings import PLOT_PATH

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

FEATURE_GROUP_MAP = feature_set_list.all_groups()
HUMAN_READABLE_MAP = feature_set_list.human_readable_map()
_COLORMAP = None
sns.set_style('whitegrid')


# Turns hue group into a color, consistent across plots
def colormap():
    global _COLORMAP
    if _COLORMAP is None:
        classifiers       = models.CLASSIFIER_KEYS
        standard_features = models.FEATURE_SETS
        new_features = models.NEW_FEATURE_SETS
        metrics      = models.METRICS

        p1 = sns.xkcd_palette(["brick", "steel blue", "moss green", "dusty rose", "grape", "pale orange", "deep pink"]) 
        p2 = sns.xkcd_palette(["purple", "green", "blue", "pink", "brown", "light blue", "grey", "orange", "tan"]) 
        p3 = sns.husl_palette(len(new_features), h=0.3)
        p4 = sns.xkcd_palette(["windows blue", "faded green", "dusty purple"])

        # p2 = sns.color_palette("husl", len(featuresets))
        MAP1 = {key: color for key, color in zip(classifiers, p1)}
        MAP2 = {key: color for key, color in zip(standard_features, p2)}
        MAP3 = {key: color for key, color in zip(new_features, p3)}
        MAP4 = {key: color for key, color in zip(metrics, p4)}

        _COLORMAP = MAP1
        _COLORMAP.update(MAP2)
        _COLORMAP.update(MAP3)
        _COLORMAP.update(MAP4)

    return _COLORMAP


def bar_plot(dfs, figname, **kwargs):
    x_col      = kwargs.pop('x_col', None)
    y_col      = kwargs.pop('y_col', None)
    hue_col    = kwargs.pop('hue_col', None)
    x_label    = kwargs.pop('x_label', "")
    y_label    = kwargs.pop('y_label', "")
    fontsize   = kwargs.pop('fontsize', 10)
    titlesize  = kwargs.pop('titlesize', 16)
    y_lim      = kwargs.pop('y_lim', (0, 1))
    show       = kwargs.pop('show', False)
    order      = kwargs.pop('order', None)
    dodge      = kwargs.pop('dodge', True)
    figsize    = kwargs.pop('figsize', (10, 8))
    font_scale = kwargs.pop('font_scale', 0.8)
    labelsize  = kwargs.pop('labelsize', None)
    title      = kwargs.pop('title', "")
    errwidth   = kwargs.pop('errwidth', 0.75)
    rotation   = kwargs.pop('rotation', None)
    capsize    = kwargs.pop('capsize', 0.2)

    if x_col is None:
        raise ValueError("No x_column entered")

    if y_col is None:
        raise ValueError("No y_column entered")

    # fig, ax = plt.subplots(figsize=figsize)
    sns.set_style('whitegrid')
    sns.set(font_scale=font_scale)
    plt.figure(figsize=figsize)
    if rotation is not None:
        plt.xticks(rotation=rotation)
    ax = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=dfs, order=order, palette=colormap(), dodge=dodge, ci=90, errwidth=errwidth, capsize=capsize)
    fig = ax.get_figure()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)
    fig.suptitle(title, fontsize=titlesize)
    fig.tight_layout()
    if y_lim is not None:
        plt.ylim(*y_lim)
    if show:
        plt.show()
    else:
        fig.savefig(PLOT_PATH + figname)


def feature_selection_plot(dfs, metric, figname='feature_selection.png', show=False):
    plt.figure(figsize=(12, 8))
    ax = sns.tsplot(data=dfs, time="number_of_features", value=metric, color=colormap(), unit="folds", condition='model')
    fig = ax.get_figure()
    ax.figure.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(PLOT_PATH + figname)


def print_ci_from_df(df, model, metric):
    fm_ci = st.t.interval(0.90, len(df) - 1, loc=np.mean(df), scale=st.sem(df))
    mean = df.mean()
    print '%s: (%s of %0.3f, and 90%% CI=%0.3f-%0.3f)' % (model, metric, mean, fm_ci[0], fm_ci[1])
    # return mean, fm_ci[0], fm_ci[1]


def print_stats(table_name):
    table = pd.read_sql_table(table_name, cnx, index_col='index')
    max_k = table.mean().argmax()
    print '%s:\n %f +/- %f, max_k: %s' % (table_name, table[max_k].mean(), table[max_k].std(), max_k)


def get_max_fold(df):
    max_k = df.mean().argmax()
    df = df[max_k].to_frame()
    df.columns = ['folds']
    df['max_k'] = int(max_k)
    return df.reset_index(drop=True)


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


def new_features_dataset_helper(key, polynomial_terms=True):
    to_drop = feature_set_list.new_features()
    # to_drop += ["count_ls_rs_switches"]
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

    if polynomial_terms:
        return data_handler.get_data(drop_features=to_drop, polynomial_terms=new_feature_set)
    else:
        return data_handler.get_data(drop_features=to_drop)


def new_feature_ablation_dataset_helper(key):
    to_drop = feature_set_list.new_features()
    # to_drop += ["count_ls_rs_switches"]
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


def map_feature_to_group(feature):
    for group_name in FEATURE_GROUP_MAP:
        if feature in FEATURE_GROUP_MAP[group_name]:
            return group_name

# Change some features, allow rest to pass through
def make_human_readable_features(feature):
    if feature in HUMAN_READABLE_MAP:
        return HUMAN_READABLE_MAP[feature]
    else:
        return feature

def get_number_of_features_in_group(group):
    return len(FEATURE_GROUP_MAP[group])


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
