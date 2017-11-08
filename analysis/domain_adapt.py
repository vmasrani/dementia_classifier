import pandas as pd
from sqlalchemy import types
from cross_validators import DomainAdaptationCV
from dementia_classifier.analysis import data_handler
from dementia_classifier.settings import DOMAIN_ADAPTATION_RESULTS_PREFIX
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


def save_domain_adapt_results_to_sql():
    Xt, yt, Xs, ys = data_handler.get_target_source_data()
    for model in models.CLASSIFIERS:
        print 'Running %s' % model
        da = DomainAdaptationCV(models.CLASSIFIERS_NEW[model], Xt, yt, Xs, ys)
        da.train_all()
        save_domain_adapt_to_sql_helper(da, model)

    da.train_majority_class()
    save_majority_class(da)


def save_majority_class(da):
    for metric in ['fms', 'acc']:
        results = da.results['majority_class'][metric]
        df = pd.DataFrame(results)
        name = "cv_majority_class_%s" % metric
        df.to_sql(name, cnx, if_exists='replace')


def save_domain_adapt_to_sql_helper(da, model_name, if_exists='replace'):
    dfs = []
    name = "%s_%s" % (DOMAIN_ADAPTATION_RESULTS_PREFIX, model_name)
    for method in da.methods:
        k_range = da.best_k[method]['k_range']
        # for metric in ['roc', 'fms', 'acc']:
        for metric in ['fms', 'acc']:
            if metric in da.results[method].keys():
                results = da.results[method][metric]
                df = pd.DataFrame(results, columns=k_range)
                df['metric'] = metric.decode('utf-8', 'ignore')
                df['method'] = method.decode('utf-8', 'ignore')
                dfs += [df]

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['method'] = types.NVARCHAR(length=255)
    df.to_sql(name, cnx, if_exists=if_exists, dtype=typedict)


# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================

# Returns 10 folds for best k (k= number of features)
def get_da_results(classifier_name, domain_adapt_method, metric):
    name = "results_domain_adaptation_%s" % classifier_name
    table = pd.read_sql_table(name, cnx, index_col='index')
    table = table[(table.metric == metric) & (table.method == domain_adapt_method)].dropna(axis=1)
    df = util.get_max_fold(table)
    df['model'] = classifier_name
    df['method'] = domain_adapt_method
    return df


# =================================================================
# ------------------------- Make plots ----------------------------
# =================================================================

def domain_adaptation_plot_helper(classifiers, metric='acc'):
    METHODS = ['target_only', 'source_only', 'relabeled', 'augment', 'coral']
    dfs = []
    for method in METHODS:
        for classifier in classifiers:
            df = get_da_results(classifier, method, metric)
            util.print_ci_from_df(df['folds'], method, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)

    plot_specs = {
        'x_col': 'method',
        'y_col': 'folds',
        'hue_col': 'model',
        'x_label': 'Model',
        'figsize': (10, 8),
        'y_label': metric,
        'y_lim': (0, 1)
    }

    figname = 'domain_adapt_plot_%s_%s.png' % (metric, classifiers[1])
    bar_plot(dfs, figname, **plot_specs)


def good_classifiers_plot(metric='acc'):
    domain_adaptation_plot_helper(models.CLASSIFIER_SET_1, metric)


def bad_classifiers_plot(metric='acc'):
    domain_adaptation_plot_helper(models.CLASSIFIER_SET_2, metric)
