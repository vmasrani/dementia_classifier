import pandas as pd
from sqlalchemy import types
from cross_validators import DementiaCV
from dementia_classifier.settings import ABLATION_RESULTS_PREFIX, NEW_FEATURES_RESULTS_PREFIX
import util
import models
from util import bar_plot

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================

# Ablation study


def save_ablation_results_to_sql():
    classifiers   = models.CLASSIFIERS
    ablation_sets = models.FEATURE_SETS
    for ablation_set in ablation_sets:
        print 'Ablating: %s' % ablation_set
        X, y, labels = util.ablation_dataset_helper(ablation_set)
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_models_to_sql_helper(trained_models, ablation_set, ABLATION_RESULTS_PREFIX)


def save_new_feature_results_to_sql():
    new_feature_set = models.NEW_FEATURE_SETS
    new_feature_set.append('none')
    classifiers  = models.CLASSIFIERS

    for feature_set in new_feature_set:
        print 'Saving new feature: %s' % feature_set
        X, y, labels = util.new_features_dataset_helper(feature_set)
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_models_to_sql_helper(trained_models, feature_set, NEW_FEATURES_RESULTS_PREFIX)


def save_models_to_sql_helper(trained_models, ablation_set, prefix, if_exists='replace'):
    method = 'default'
    dfs = []
    for model in trained_models:
        cv = trained_models[model]
        k_range = cv.best_k[method]['k_range']
        for metric in models.METRICS:
            if metric in cv.results[method].keys():
                results = cv.results[method][metric]
                df = pd.DataFrame(results, columns=k_range)
                df['metric'] = metric.decode('utf-8', 'ignore')
                df['model'] = model
                dfs += [df]

    name = "%s_%s" % (prefix, ablation_set)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)
    df.to_sql(name, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================

def get_ablation_results(ablation_set, model, metric, results_table_suffix='results_ablation'):
    reference = "results_new_features_none"
    ref = pd.read_sql_table(reference, cnx, index_col='index')
    ref = ref[(ref.metric == metric) & (ref.model == model)].dropna(axis=1)
    max_ref_k = ref.mean().argmax()
    ref = ref[max_ref_k].to_frame().reset_index(drop=True)
    ref.columns = ['folds']

    # abl == ablated
    name = "%s_%s" % (results_table_suffix, ablation_set)
    abl = pd.read_sql_table(name, cnx, index_col='index')
    abl = abl[(abl.metric == metric) & (abl.model == model)].dropna(axis=1)

    max_abl_k = abl.mean().argmax()
    abl = abl[max_abl_k].to_frame().reset_index(drop=True)
    abl.columns = ['folds']

    # Difference from reference
    diff = abl['folds'] - ref['folds']
    diff = diff.to_frame()
    diff = diff * 100.0
    diff['model'] = model
    diff['metric'] = metric

    ablation_set_name = "%s (%i)" % (ablation_set.title(), util.get_number_of_features_in_group(ablation_set))

    diff['ablation_set'] = ablation_set_name

    return diff


def get_new_feature_results(new_feature_set, model, metric, absolute=True):
    reference = "results_new_features_none"
    ref = pd.read_sql_table(reference, cnx, index_col='index')
    ref = ref[(ref.metric == metric) & (ref.model == model)].dropna(axis=1)

    max_ref_k = ref.mean().argmax()
    ref = ref[max_ref_k].to_frame().reset_index(drop=True)
    ref.columns = ['folds']

    # nfs == new feature set
    name = "results_new_features_%s" % new_feature_set
    nfs = pd.read_sql_table(name, cnx, index_col='index')
    nfs = nfs[(nfs.metric == metric) & (nfs.model == model)].dropna(axis=1)
    max_nfs_k = nfs.mean().argmax()
    nfs = nfs[max_nfs_k].to_frame().reset_index(drop=True)
    nfs.columns = ['folds']

    if not absolute:
        nfs = nfs - ref
        nfs = nfs * 100.0
    

    nfs.columns = ['folds']
    nfs['model'] = model
    nfs['metric'] = metric
    nfs['new_feature_set'] = new_feature_set

    return nfs


# =================================================================
# ------------------------- Make plots ----------------------------
# ================================================================


def ablation_plot(metric='acc'):
    classifiers = list(models.CLASSIFIER_KEYS)
    ablation_sets = models.FEATURE_SETS

    classifiers.remove('DummyClassifier')

    dfs = []

    for ab_set in ablation_sets:
        for classifier in classifiers:

            df = get_ablation_results(ab_set, classifier, metric)
            util.print_ci_from_df(df['folds'], ab_set, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)

    plot_specs = {
        'x_col': 'ablation_set',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'Feature Set',
        'y_label': "%% change in %s" % metric,
        'figsize': (10, 8),
        'fontsize': 14,
        'y_lim': None,
        'errwidth': 0.75,
        'labelsize': 10,
        'rotation': 15
    }

    figname = 'ablation_plot_%s.png' % metric

    bar_plot(dfs, figname, **plot_specs)


def new_feature_set_plot(metric='acc', absolute=True):
    classifiers = list(models.CLASSIFIER_KEYS)
    new_features = []
    if absolute:
        new_features += ['none']
    new_features += models.NEW_FEATURE_SETS

    classifiers.remove('DummyClassifier')
    
    dfs = []
    for fs in new_features:
        for classifier in classifiers:
            df = get_new_feature_results(fs, classifier, metric, absolute=absolute)
            util.print_ci_from_df(df['folds'], fs, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)

    if metric == 'acc':
        y_label = "Accuracy"
    elif metric == 'fms':
        y_label = "F-Measure"
    else:
        y_label = "AUC"

    y_lim = (0.68, .90)
    figname = 'new_feature_plot_%s' % metric

    if not absolute:
        y_label = "%% change in %s" % y_label
        y_lim = None
        figname = figname + '_relative'

    figname = figname + '.png'

    plot_specs = {
        'x_col': 'new_feature_set',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'feature_set',
        'y_label': y_label,
        'y_lim': y_lim,
        'figsize': (10, 8),
    }

    bar_plot(dfs, figname, **plot_specs)
