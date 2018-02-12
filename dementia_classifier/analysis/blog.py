import pandas as pd
from sqlalchemy import types
import seaborn as sns
from cross_validators import BlogCV
from dementia_classifier.feature_extraction.feature_sets import feature_set_list
from dementia_classifier.analysis import data_handler
from dementia_classifier.analysis.feature_set import save_models_to_sql_helper, get_ablation_results
from dementia_classifier.settings import BLOG_RESULTS, BLOG_ABLATION_PREFIX, SQL_BLOG_SUFFIX
from dementia_classifier.analysis.general_plots import get_feature_selection_curve
import util
import models
from util import bar_plot
from util import feature_selection_plot
from dementia_classifier.settings import PLOT_PATH

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

# =================================================================
# ----------------------Save results to sql------------------------
# =================================================================


def save_blog_results_to_sql():
    X, y, labels = data_handler.get_blog_data()
    classifiers = models.CLASSIFIERS
    trained_models = {model: BlogCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
    save_blogs_to_sql_helper(trained_models, if_exists='append')


def save_blogs_to_sql_helper(trained_models, if_exists='replace'):
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

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)

    df.to_sql(BLOG_RESULTS, cnx, if_exists=if_exists, dtype=typedict)


def save_blog_ablation_results_to_sql():
    classifiers   = models.CLASSIFIERS
    ablation_sets = models.BLOG_FEATURE_SETS
    for ablation_set in ablation_sets:
        print 'Ablating: %s' % ablation_set
        X, y, labels = util.blog_feature_ablation(ablation_set)
        trained_models = {model: BlogCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_models_to_sql_helper(trained_models, ablation_set, prefix=BLOG_ABLATION_PREFIX)

# =================================================================
# ----------------------Get results from sql-----------------------
# =================================================================


def get_blog_results(model, metric):
    name = "results_blog"
    table = pd.read_sql_table(name, cnx, index_col='index')
    table = table[(table.metric == metric) & (table.model == model)].dropna(axis=1)
    max_k = table.mean().argmax()
    df = table[max_k].to_frame()
    df.columns = ['folds']
    df['model'] = model
    df['metric'] = metric
    return df


def get_blog_feature_rankings():
    X, y, labels = data_handler.get_blog_data()
    cv = BlogCV(None, X=X, y=y, labels=labels)
    feature_ranks = cv.feature_rank()
    feature_ranks['group'] = feature_ranks['feature'].apply(util.map_feature_to_group)
    feature_ranks['feature'] = feature_ranks['feature'].apply(util.make_human_readable_features)
    return feature_ranks


def get_blog_feature(feature):
    dfs = []
    for blog in models.BLOG_NAMES:
        name = "%s_%s" % (blog, SQL_BLOG_SUFFIX)
        table = pd.read_sql_table(name, cnx)
        df = pd.DataFrame(table[feature].astype(float))
        df['blog'] = blog
        dfs.append(df)

    return pd.concat(dfs)
# =================================================================
# ------------------------- Make plots ----------------------------
# =================================================================


def blog_plot():
    print "Plotting blog_plot"
    metrics = models.METRICS
    dfs = []
    for classifier in models.CLASSIFIER_KEYS:
        for metric in metrics:
            df = get_blog_results(classifier, metric)
            util.print_ci_from_df(df['folds'], classifier, metric)
            dfs.append(df)

    dfs = pd.concat(dfs)
    
    plot_specs = {
        'x_col': 'model',
        'y_col': 'folds',
        'hue_col': 'metric',
        'x_label': 'Model',
        'y_label': 'Performance',
        'font_scale': 1.2,
        'fontsize': 20,
        'rotation': 15
    }

    figname = 'blog_plot.pdf'

    bar_plot(dfs, figname, **plot_specs)


def plot_blog_feature_selection_curve(show=False, metric='fms'):
    print "Plotting plot_blog_feature_selection_curve, metric: %s" % metric
    firsthalf = ['DummyClassifier', 'LogReg', 'KNeighbors']
    secondhalf = ['DummyClassifier', 'RandomForest', 'GausNaiveBayes', 'SVC']

    if metric == 'acc':
        y_label = "Accuracy"
    elif metric == 'fms':
        y_label = "F-Measure"
    else:
        y_label = "AUC"
   
    plot_specs = {
        "title": "Feature Selection Curve",
        'x_label': 'Number of Features',
        'y_label': y_label,
    }

    for idx, half in enumerate((firsthalf, secondhalf)):
        dfs = []
        for classifier in half:
            df = get_feature_selection_curve(classifier, metric, table_name='results_blog')
            dfs.append(df)
        dfs = pd.concat(dfs)
        if idx == 1:
            plot_specs['title'] = ""
        figname = 'blog_feature_selection_%s_%i.pdf' % (metric, idx)
        feature_selection_plot(dfs, metric, figname=figname, show=show, **plot_specs)


def blog_feature_box_plot(feature, trim_zeros=False, y_lim=None):
    print "Plotting blog_feature_box_plot, feature %s" % feature
    df = get_blog_feature(feature)
    if trim_zeros:
        df = df[df[feature] > 0]
    
    figname = 'blog_boxplot_%s.pdf' % feature

    dem_blogs = [
        "living-with-alzhiemers",
        "parkblog-silverfox",
        "creatingmemories"
    ]
    
    ctrl_blogs = [
        "journeywithdementia",
        "helpparentsagewell",
        "earlyonset",
    ]
    
    pal = sns.xkcd_palette(["brick", "steel blue"])
    my_pal = {blog: pal[0] if blog in dem_blogs else pal[1] for blog in dem_blogs + ctrl_blogs}

    plot_specs = {
        'x': 'blog',
        'y': feature,
        'palette': my_pal,
        'y_lim': y_lim,
        'fontsize': 30,
        'labelsize': 20,
    }

    util.box_plot(df, figname, **plot_specs)


def blog_ablation_plot(metric='acc'):
    print "Plotting blog_ablation_plot, metric: %s" % metric
    classifiers = models.CLASSIFIER_KEYS
    ablation_sets = models.BLOG_FEATURE_SETS

    classifiers.remove('DummyClassifier')
    
    dfs = []

    for classifier in classifiers:
        for ab_set in ablation_sets:
            df = get_ablation_results(ab_set, classifier, metric, BLOG_ABLATION_PREFIX)
            util.print_ci_from_df(df['folds'], classifier, metric)
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

    figname = 'blog_ablation_plot.pdf'

    bar_plot(dfs, figname, **plot_specs)


def plot_blog_feature_rank(show=False):
    print "Plotting plot_blog_feature_rank"
    dfs = get_blog_feature_rankings()
    order = dfs[['feature', 'weight', 'group']].groupby(['feature', 'group'])['weight'].mean().reset_index().sort_values(['group', 'weight'], ascending=[True, False])['feature']
    plot_specs = {
        'x_col': 'weight',
        'y_col': 'feature',
        'hue_col': 'group',
        'x_label': 'Feature Score',
        'y_label': 'Feature Sets',
        'order': order,
        'dodge': False,
        'labelsize': 8,
        'figsize': (8, 11),
        'show': show,
        'y_lim': None,
        'capsize': .2,

    }

    figname = 'blog_feature_rank.pdf'
    bar_plot(dfs, figname, **plot_specs)
