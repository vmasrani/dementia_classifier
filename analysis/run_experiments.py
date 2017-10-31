import pandas as pd
from sqlalchemy import types
import util
from cross_validators import DementiaCV, DomainAdaptationCV, BlogCV
from dementia_classifier.analysis import data_handler


# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------


# =================================================================
# ----------------------Domain Adaptation--------------------------
# =================================================================


def save_majority_class(da):
    for metric in ['fms', 'acc']:
        results = da.results['majority_class'][metric]
        df = pd.DataFrame(results)
        name = "cv_majority_class_%s" % metric
        df.to_sql(name, cnx, if_exists='replace')


def save_domain_adapt_to_sql(da, model_name):
    dfs = []
    name = "results_domain_adaptation_%s_10fold" % (model_name)
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
    df.to_sql(name, cnx, if_exists='replace', dtype=typedict)


def domain_adapt():
    Xt, yt, Xs, ys = data_handler.get_target_source_data()
    for model in util.CLASSIFIERS:
        da = DomainAdaptationCV(util.CLASSIFIERS[model], Xt, yt, Xs, ys)
        da.train_all()
        save_domain_adapt_to_sql(da, model)

    da.train_majority_class()
    save_majority_class(da)


# =================================================================
# -------------------------- Abalation ----------------------------
# =================================================================

def save_ablation_to_sql(trained_models, ablation_set, results_suffix='results_ablation'):
    method = 'default'
    dfs = []
    for model in trained_models:
        cv = trained_models[model]
        k_range = cv.best_k[method]['k_range']
        for metric in ['roc', 'fms', 'acc']:
            if metric in cv.results[method].keys():
                results = cv.results[method][metric]
                df = pd.DataFrame(results, columns=k_range)
                df['metric'] = metric.decode('utf-8', 'ignore')
                df['model'] = model
                dfs += [df]
                name = "%s_%s" % (results_suffix, ablation_set)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    typedict = {col_name: types.Float(precision=5, asdecimal=True) for col_name in df}
    typedict['metric'] = types.NVARCHAR(length=255)
    typedict['model']  = types.NVARCHAR(length=255)
    df.to_sql(name, cnx, if_exists='replace', dtype=typedict)


def ablation():
    classifiers   = util.CLASSIFIERS
    ablation_sets = util.ABLATION_SETS
    for ablation_set in ablation_sets:
        print 'Ablating: %s' % ablation_set
        X, y, labels = util.ablation_dataset_helper(ablation_set)
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_ablation_to_sql(trained_models, ablation_set)


# =================================================================
# ------------------------ New Feature-----------------------------
# =================================================================

def save_new_features_to_sql(trained_models, new_feature_set):
    method = 'default'
    dfs = []
    for model in trained_models:
        cv = trained_models[model]
        k_range = cv.best_k[method]['k_range']
        for metric in ['roc', 'fms', 'acc']:
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

    name = "results_new_features_%s" % new_feature_set
    df.to_sql(name, cnx, if_exists='replace', dtype=typedict)


def new_features():
    new_feature_set = util.NEW_FEATURE_SETS
    classifiers  = util.CLASSIFIERS
    for feature_set in new_feature_set:
        print 'Saving new feature: %s' % feature_set
        X, y, labels = util.new_features_dataset_helper(feature_set)
        trained_models = {model: DementiaCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_new_features_to_sql(trained_models, feature_set)


# def run_learning_curve(with_disc):
#     model = LogisticRegression(penalty='l2', C=REGULARIZATION_CONSTANT)
#     k_range = range(50, 900, 1)
#     percentages = [.25, 0.5, 0.75, 1.0]

#     for t in np.linspace(1, 15, 14):
#         t = int(t)
#         trials = []
#         bestk  = []
#         print "running trial %i" % t
#         for p in percentages:
#             print "percentage: %f" % p
#             da = DomainAdapter(model, f1_score, with_disc=with_disc, silent=False, trim=p, random_state=t)
#             da.train_model('augment', k_range)
#             trials.append(da.best_score['augment'])
#             bestk.append(da.best_k['augment'])

#         trials = pd.DataFrame([trials], columns=percentages, index=[t])
#         bestk = pd.DataFrame([bestk], columns=percentages, index=[t])

#         if with_disc:
#             name = "learning_curve_with_disc_target_source_trimmed"
#         else:
#             name = "learning_curve_without_disc_target_source_trimmed"

#         trials.to_sql(name, cnx, if_exists='append')
#         bestk.to_sql(name + '_bestk', cnx, if_exists='append')


# =================================================================
# ----------------------Blog analysis------------------------------
# =================================================================

def blogs():
    X, y, labels = data_handler.get_blog_data()
    classifiers = util.CLASSIFIERS
    trained_models = {model: BlogCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
    save_blogs(trained_models)


def save_blogs(trained_models):
    method = 'default'
    dfs = []
    for model in trained_models:
        cv = trained_models[model]
        k_range = cv.best_k[method]['k_range']
        for metric in ['roc', 'fms', 'acc']:
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

    name = "results_blog"
    df.to_sql(name, cnx, if_exists='replace', dtype=typedict)


def blog_ablation():
    classifiers   = util.CLASSIFIERS
    ablation_sets = util.BLOG_ABLATION_SETS
    results_suffix = 'results_blog_ablation'
    
    for ablation_set in ablation_sets:
        print 'Ablating: %s' % ablation_set
        X, y, labels = util.blog_feature_ablation(ablation_set)
        trained_models = {model: BlogCV(classifiers[model], X=X, y=y, labels=labels).train_model('default') for model in classifiers}
        save_ablation_to_sql(trained_models, ablation_set, results_suffix=results_suffix)
