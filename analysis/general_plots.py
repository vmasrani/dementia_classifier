import pandas as pd
from sqlalchemy import types
import matplotlib.pyplot as plt
from cross_validators import DementiaCV, BlogCV
from dementia_classifier.settings import ABLATION_RESULTS_PREFIX, NEW_FEATURES_RESULTS_PREFIX
from dementia_classifier.analysis import data_handler
import seaborn as sns
import util
import models
from util import bar_plot, new_features_dataset_helper, feature_selection_plot
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.settings import PLOT_PATH
# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------


# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================

# Ablation study

def save_model_to_sql(model_name):
    model = util.CLASSIFIERS_NEW[model_name]
    X, y, labels = util.ablation_dataset_helper('none')
    cv = DementiaCV(model, X=X, y=y, labels=labels).train_model('default', k_range=range(1, 150))
    save_model_to_sql_helper(cv, model_name, '%s_standard_features' % model_name)


def save_model_to_sql_helper(cv, model_name, sqlname, if_exists='replace'):
    method = 'default'
    dfs = []
    k_range = cv.best_k[method]['k_range']
    for metric in models.METRICS:
        if metric in cv.results[method].keys():
            results = cv.results[method][metric]
            df = pd.DataFrame(results, columns=k_range)
            df['metric'] = metric.decode('utf-8', 'ignore')
            df['model'] = model_name
            dfs += [df]

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)
    df.to_sql(sqlname, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================


def get_vanilla_results(model, metric):
    table = "results_new_features_none"

    df = pd.read_sql_table(table, cnx, index_col='index')
    df = df[(df.metric == metric) & (df.model == model)].dropna(axis=1)

    max_k = df.mean().argmax()
    df = df[max_k].to_frame().reset_index(drop=True)
    df.columns = ['folds']

    df['model'] = model
    df['metric'] = metric

    return df


def get_feature_rankings(dataset='halves', polynomial_terms=False):
    X, y, labels = new_features_dataset_helper(dataset, polynomial_terms)
    cv = DementiaCV(None, X, y, labels)
    feature_ranks = cv.feature_rank()
    feature_ranks['group'] = feature_ranks['feature'].apply(util.map_feature_to_group)
    feature_ranks['feature'] = feature_ranks['feature'].apply(util.make_human_readable_features)
    return feature_ranks




def get_feature_selection_curve(model, metric, table_name="results_new_features_none"):
    df = pd.read_sql_table(table_name, cnx, index_col='index')
    df = df[(df.metric == metric) & (df.model == model)].dropna(axis=1)
    # First reset_index(drop=True) sets index to 0-nfolds, second saves as column 'folds'
    df = df.drop(['metric', 'model'], axis=1).reset_index(drop=True)
    print "%s_%s" % (model, table_name)
    print df.mean().max()
    df.index.name = 'folds'
    df = df.reset_index().melt(id_vars='folds', var_name='number_of_features', value_name=metric)
    df['number_of_features'] = df['number_of_features'].astype(float)
    df['model'] = model
    return df

# =================================================================
# ------------------------- Make plots ----------------------------
# ================================================================


def vanilla_feature_set_plot(show=False):
    metrics = ["acc", "fms", "roc"]
    dfs = []
    classifiers = models.CLASSIFIER_KEYS
    for classifier in classifiers:
        for metric in metrics:
            df = get_vanilla_results(classifier, metric)
            util.print_ci_from_df(df['folds'], classifier, metric)
            dfs.append(df)

    dfs = pd.concat(dfs)

    plot_specs = {
        'x_col': 'model',
        'y_col': 'folds',
        'hue_col': 'metric',
        'x_label': 'Model',
        'y_label': 'Metric',
        'figsize': (12, 10),
        'y_lim': None,
        'show': show,
        'title': "10-Fold Cross Validation Performance"
    }

    figname = 'vanilla_results.png'
    bar_plot(dfs, figname, **plot_specs)


def plot_feature_rank(dataset, show=False):
    dfs = get_feature_rankings(dataset=dataset, polynomial_terms=False)
    order = dfs[['feature', 'group']].drop_duplicates().sort_values('group')['feature']

    plot_specs = {
        'x_col': 'weight',
        'y_col': 'feature',
        'hue_col': 'group',
        'x_label': 'Feature Score',
        'y_label': 'Feature Sets',
        'order': order,
        'dodge': False,
        'labelsize': 5,
        'show': show,
        'y_lim': None,
        'capsize': .2,

    }

    figname = 'feature_rank_%s.png' % dataset
    bar_plot(dfs, figname, **plot_specs)


def plot_feature_selection_curve(show=False, metric='fms'):
    dfs = []
    classifiers = models.CLASSIFIER_KEYS
    for classifier in classifiers:
        df = get_feature_selection_curve(classifier, metric)
        dfs.append(df)
    dfs = pd.concat(dfs)
    feature_selection_plot(dfs, metric, show=show)