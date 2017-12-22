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


def save_new_feature_results_to_sql(polynomial_terms=True):
    new_feature_set = models.NEW_FEATURE_SETS
    new_feature_set.append('none')
    classifiers  = models.CLASSIFIERS

    prefix = NEW_FEATURES_RESULTS_PREFIX
    
    if polynomial_terms:
        prefix = NEW_FEATURES_RESULTS_PREFIX + "_poly"

    for feature_set in new_feature_set:
        print 'Saving new feature: %s' % feature_set
        X, y, labels = util.new_features_dataset_helper(feature_set, polynomial_terms=polynomial_terms)
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_models_to_sql_helper(trained_models, feature_set, prefix)


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


# Compare our results to fraser by removing age (but adding halves+poly)
def save_fraser_comparision():
    new_feature_set = ['none']
    # new_feature_set = ['none', 'halves']
    prefix = NEW_FEATURES_RESULTS_PREFIX + "_fraser_comparision"
    for new_feature in new_feature_set:
        X, y, labels = util.new_features_dataset_helper(new_feature, polynomial_terms=True)
        X = X.drop('age', axis=1, errors='ignore')
        classifiers = {'LogReg': models.CLASSIFIERS['LogReg']}
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_models_to_sql_helper(trained_models, new_feature, prefix)


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

    if ablation_set == 'information_content':
        ablation_set_name = "%s (%i)" % ("Info-Units", util.get_number_of_features_in_group(ablation_set))
    else:
        ablation_set_name = "%s (%i)" % (ablation_set.title(), util.get_number_of_features_in_group(ablation_set))

    diff['ablation_set'] = ablation_set_name

    return diff


def get_new_feature_results(new_feature_set, model, metric, absolute=True, poly=True):
    reference = "results_new_features_none"
    ref = pd.read_sql_table(reference, cnx, index_col='index')
    ref = ref[(ref.metric == metric) & (ref.model == model)].dropna(axis=1)

    max_ref_k = ref.mean().argmax()
    ref = ref[max_ref_k].to_frame().reset_index(drop=True)
    ref.columns = ['folds']

    # nfs == new feature set
    if new_feature_set == 'halves' and poly:
        name = "results_new_features_poly_%s" % new_feature_set
    else:
        name = "results_new_features_%s" % new_feature_set
    nfs = pd.read_sql_table(name, cnx, index_col='index')
    nfs = nfs[(nfs.metric == metric) & (nfs.model == model)].dropna(axis=1)
    max_nfs_k = nfs.mean().argmax()
    nfs = nfs[max_nfs_k].to_frame().reset_index(drop=True)
    nfs.columns = ['folds']

    if not absolute:
        nfs = nfs - ref

    nfs.columns = ['folds']
    nfs['model'] = model
    nfs['metric'] = metric
    nfs['new_feature_set'] = new_feature_set

    return nfs


def get_feature(feature):
    X, y, _ = util.new_features_dataset_helper('halves', polynomial_terms=False)
    if feature not in X.columns:
        raise ValueError('Could not find %s in %s' % (feature, X.columns))
    y.name = 'Dementia'
    return pd.concat([X[feature], y], axis=1)


# =================================================================
# ------------------------- Make plots ----------------------------
# ================================================================


def ablation_plot(metric='acc'):
    print "Plotting ablation_plot, metric %s" % metric
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

    human_readable = {"acc": "Accuracy", "fms": "F-Measure", "roc": "AUC"}

    plot_specs = {
        'x_col': 'ablation_set',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'Feature Set',
        'y_label': "Change in %s " % human_readable[metric],
        'title': "Feature Ablation",
        'figsize': (10, 8),
        'fontsize': 20,
        'font_scale': 1.2,
        'y_lim': None,
        'errwidth': 0.75,
        'labelsize': 10,
        'rotation': 15
    }
    
    figname = 'ablation_plot_%s.pdf' % metric

    bar_plot(dfs, figname, **plot_specs)


def print_fraser_comparision(metric='fms'):
    print 'metric %s' % metric
    t0 = 'results_new_features_fraser_comparision_none'
    t1 = 'results_new_features_fraser_comparision_halves'
    t2 = 'results_new_features_poly_halves'
    t3 = 'results_new_features_none'
    df0 = util.get_max_fold_from_table(metric, t0)
    df1 = util.get_max_fold_from_table(metric, t1)
    df2 = util.get_max_fold_from_table(metric, t2)
    df3 = util.get_max_fold_from_table(metric, t3)
    util.print_ci_from_df(df0['folds'], "without_age_without_halves", "LogReg")
    util.print_ci_from_df(df1['folds'], "without_age_with_halves", "LogReg")
    util.print_ci_from_df(df2['folds'], "with_age_with_halves", "LogReg")
    util.print_ci_from_df(df3['folds'], "with_age_without_halves", "LogReg")


def new_feature_set_plot(metric='acc', absolute=True, poly=True, show=False):
    print "Plotting new_feature_set_plot, metric: %s" % metric
    classifiers = list(models.CLASSIFIER_KEYS)
    new_features = []
    if absolute:
        new_features += ['none']
    new_features += models.NEW_FEATURE_SETS
    classifiers.remove('DummyClassifier')
    dfs = []

    for fs in new_features:
        for classifier in classifiers:
            df = get_new_feature_results(fs, classifier, metric, absolute=absolute, poly=poly)
            util.print_ci_from_df(df['folds'], fs, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)
    dfs = dfs.replace('none', 'baseline')

    y_lim = (.68, .90)

    if metric == 'acc':
        y_label = "Accuracy"
    elif metric == 'fms':
        y_label = "F-Measure"
    else:
        y_label = "AUC"
        y_lim = (.70, .95)

    figname = 'new_feature_plot_%s' % metric
    title = 'Performance w/ New Feature Sets'
    if not absolute:
        y_label = "Change in %s" % y_label
        y_lim = (-.10, .10)
        figname = figname + '_relative'
        title = 'Change in Performance w/ New Feature Sets'

    plot_specs = {
        'x_col': 'new_feature_set',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'Feature Set',
        'y_label': y_label,
        'y_lim': y_lim,
        'figsize': (10, 8),
        'fontsize': 20,
        'font_scale': 1.2,
        'labelsize': 15,
        'show': show,
        'title': title,
    }
    
    # We use polynomial terms as well for halves
    if poly:
        dfs = dfs.replace('halves', 'halves+quadratic')
    else:
        figname = figname + '_without_quadratic'
    
    figname = figname + '.pdf'
    bar_plot(dfs, figname, **plot_specs)


def feature_box_plot(feature):
    print "Plotting feature_box_plot, feature %s" % feature
    df = get_feature(feature)
    
    figname = 'dbank_boxplot_%s.pdf' % feature
    
    plot_specs = {
        'x': 'Dementia',
        'y': feature,
    }
    
    util.box_plot(df, figname, **plot_specs)
