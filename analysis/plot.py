import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import util
from data_handler import get_da_results, get_ablation_results, get_new_feature_results, get_blog_results

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

# Table names
# CLASSIFIERS = [
#     "LogReg",
#     "RandomForest",
#     "MLP",
# ]


# LEARNING_CURVE         = 'learning_curve_with_disc_target_source_trimmed'
# LEARNING_CURVE_WITHOUT = 'learning_curve_without_disc_target_source_trimmed'


# Standard formatting for all barplots
def bar_plot(dfs, x_col=None, y_col=None, hue_col=None, x_label=None, y_label=None):

    sns.set_style('whitegrid')
    # sns.set(font_scale=1.8)
    # sns.plt.ylim(0, 1)
    ax = sns.barplot(x=x_col, y=y_col, hue=hue_col, data=dfs, ci=90, errwidth=1.25, capsize=.2)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.figure.tight_layout()
    sns.plt.show()


def print_stats(table_name):
    table = pd.read_sql_table(table_name, cnx, index_col='index')
    max_k = table.mean().argmax()
    print '%s:\n %f +/- %f, max_k: %s' % (table_name, table[max_k].mean(), table[max_k].std(), max_k)


def print_ci_from_df(df, model, metric):
    fm_ci = st.t.interval(0.90, len(df) - 1, loc=np.mean(df), scale=st.sem(df))
    print '%s: (%s of %0.3f, and 90%% CI=%0.3f-%0.3f)' % (model, metric, df.mean(), fm_ci[0], fm_ci[1])


def get_max_fold_from_arr(arr):
    table = pd.DataFrame(arr)
    if table.shape == (1, 10):
        table = table.T
    else:
        max_k = table.mean().argmax()
        table = table[max_k].to_frame()
    table.columns = ['folds']
    return table


def domain_adaptation_plot():
    METHODS = ['target_only', 'source_only', 'relabeled', 'augment', 'coral']
    metric = 'acc'
    dfs = pd.concat([get_da_results(classifier, method, 'acc') for classifier in util.CLASSIFIERS for method in METHODS])
    
    bar_plot(dfs, x_col='method', y_col='folds', hue_col='model', x_label='Model', y_label=metric)
    

def ablation_plot():
    classifiers = util.CLASSIFIERS.keys()
    ablation_sets = util.ABLATION_SETS
    
    classifiers.remove('DummyClassifier')
    ablation_sets.remove('none')
    metric = 'fms'
    dfs = pd.concat([get_ablation_results(ab_set, classifier, metric) for classifier in classifiers for ab_set in ablation_sets])
    bar_plot(dfs, x_col='ablation_set', y_col='folds', hue_col='model', x_label='Model', y_label="%% change in %s" % metric)
    

def standard_feature_set_plot():
    metrics = ["acc", "fms", "roc"]
    dfs = []
    for classifier in util.CLASSIFIERS:
        for metric in metrics:
            df = get_new_feature_results('none', classifier, metric)
            print_ci_from_df(df['folds'], classifier, metric)
            dfs.append(df)
    
    dfs = pd.concat(dfs)
    bar_plot(dfs, 'model', 'folds', 'metric', 'model', 'performance')


def new_feature_sets():
    classifiers = util.CLASSIFIERS.keys()
    new_features = util.NEW_FEATURE_SETS
    
    classifiers.remove('DummyClassifier')
    new_features.remove('none')

    metric = 'acc'
    dfs = []
    for fs in new_features:
        for classifier in classifiers:
            df = get_new_feature_results(fs, classifier, metric)
            print_ci_from_df(df['folds'], fs, classifier)
            dfs.append(df)

    dfs = pd.concat(dfs)

    bar_plot(dfs, x_col='new_feature_set', y_col='folds', hue_col='model', x_label='feature_set', y_label="%% change in %s" % metric)
    

def blog_plot():
    metrics = ["acc", "fms", "roc"]
    dfs = []
    for classifier in util.CLASSIFIERS_KEYS:
        for metric in metrics:
            df = get_blog_results(classifier, metric)
            print_ci_from_df(df['folds'], classifier, metric)
            dfs.append(df)

    dfs = pd.concat(dfs)
    bar_plot(dfs, 'model', 'folds', 'metric', 'model', 'performance')



# def learning_curve():
#     sns.set_style('whitegrid')
#     table1 = pd.read_sql_table(LEARNING_CURVE, cnx, index_col='index')
#     table2 = pd.read_sql_table(LEARNING_CURVE_WITHOUT, cnx, index_col='index')

#     dfs1 = pd.DataFrame(table1.T.stack().reset_index(level=1, drop=True))
#     dfs2 = pd.DataFrame(table2.T.stack().reset_index(level=1, drop=True))
#     dfs1.columns = ['folds']
#     dfs2.columns = ['folds']
#     dfs1['percentage'] = dfs1.index
#     dfs2['percentage'] = dfs2.index

#     dfs1[' '] = 'With Discourse Features'
#     dfs2[' '] = 'Without Discourse Features'

#     dfs = pd.concat([dfs1,dfs2])
#     sns.set(font_scale=1.8)
#     ax = sns.factorplot(x='percentage', y='folds', hue=' ', data=dfs, ci=90, errwidth=1.25, capsize=.2)
#     ax.set(xlabel='Sample size (percentage)', ylabel='F-Measure')
#     sns.plt.show()


# def lexicalplot():
#     model = LogisticRegression(penalty='l2', C=1)
#     da_lex = LexicalOnlyCV(model, metric=accuracy_score)
#     # da_lex = DomainAdapter(model, metric=accuracy_score, lexical_only=True)
#     da_lex.train_all()
#     dfs = []
#     for model in da_lex.results.keys():
#         withdisc = get_max_fold_from_arr(da_lex.results[model])
#         withdisc['Model'] = model
#         dfs.append(withdisc)
#     dfs = pd.concat(dfs)

#     sns.set_style('whitegrid')
#     sns.set(font_scale=1.8)
#     sns.plt.ylim(0, 1)
#     ax = sns.barplot(x='Model', y='folds', data=dfs, ci=90, errwidth=1.25, capsize=.2)
#     ax.set(xlabel='Model', ylabel='Accuracy Score')
#     ax.figure.tight_layout()
#     sns.plt.show()


# def blogplot(metric):
    # names = [
    #     "Neural Net",
    # ]

    # classifiers = [
    #     MLPClassifier(alpha=1),
    # ]

    # names = [
    #     "Random",
    #     "Log_Reg",
    #     "KNN",
    #     "Decision Tree",
    #     "Random Forest",
    #     "Neural Net",
    #     "Naive Bayes"]

    # classifiers = [
    #     DummyClassifier("most_frequent"),
    #     LogisticRegression(penalty='l2', C=1),
    #     KNeighborsClassifier(3),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1),
    #     GaussianNB(),
    # ]

    # dfs = []

    # ba = BlogAdapter(classifiers[0], metric=metric, silent=True)

    # # Train all models

    # for i in range(len(names)):
    #     model = classifiers[i]
    #     name = names[i]
    #     ba.train_model(model, 'model')
    #     blog = get_max_fold_from_arr(ba.results['model'])
    #     print_ci_from_df(blog, name)
    #     blog['Model'] = name
    #     dfs.append(blog)
    # dfs = pd.concat(dfs)
    # import pdb
    # pdb.set_trace()
    # sns.set_style('whitegrid')
    # sns.set(font_scale=1.2)
    # sns.plt.ylim(0, 1)
    # ax = sns.barplot(x='Model', y='folds', data=dfs, ci=90, errwidth=1.25, capsize=.2, color='skyblue')
    # ax.set(xlabel='Model', ylabel="AUC")
    # ax.figure.tight_layout()
    # sns.plt.show()

    # log_reg_score = np.max(np.mean(ba.results['log_reg'],axis=0))
    # majority_class_score = np.max(np.mean(ba.results['majority_class'],axis=0))
    # print metric
    # print "log_reg_score: %f" % log_reg_score
    # print "majority_class_score: %f" % majority_class_score


# def blogscatter(y, save=False):
#     df = data_handler.get_blog_scatterplot_data()

#     # All this mess is to set up the labels properly :(
#     mindate = df["date"].min().year
#     maxdate = df["date"].max().year
#     yearrange = [str(year) for year in range(mindate, maxdate + 1)]
#     yearrange = [year.replace('20', '\'')for year in yearrange]
#     df["date"] = df["date"] - pd.to_datetime('2000')
#     df["date"] = df["date"].map(lambda d: d.days)
#     minday = df["date"].min()
#     maxday = df["date"].max()
#     xticks = range(minday, maxday, (maxday - minday) / (len(yearrange)))

#     sns.set(font_scale=1.8)

#     colors = {
#         'creatingmemories': '#94C9DE',
#         'living-with-alzhiemers': '#94C9DE',
#         'parkblog-silverfox': '#94C9DE',
#         'journeywithdementia': '#97C83C',
#         'earlyonset': '#97C83C',
#         'helpparentsagewell': '#97C83C'
#     }

#     plot = sns.lmplot(x="date", y=y, col="blog", hue="blog", data=df,
#                       col_wrap=3, ci=None, palette=colors, size=6,
#                       scatter_kws={"s": 50, "alpha": 1},)

#     plot = plot.set_axis_labels("Year", "SUBTL score")

#     plot.set(xticks=xticks[::2], xticklabels=yearrange[::2])
#     if save:
#         fig = plot.fig
#         fig.savefig("/Users/vmasrani/dev/Masters/Dementia/output/blogs/%s.png" % y)
#     else:
#         sns.plt.show()


# def print_blog_stats():
#     df = data_handler.get_blog_scatterplot_data()
#     for blog in df.blog.unique():
#         print "Blog: %s" % blog
#         is_dementia = df[df.blog == blog].dementia.unique()
#         print "Dementia: %s" % str(bool(is_dementia))
#         print "%d posts\n" % df[df.blog == blog].shape[0]
#     print '========================'
#     print 'Total: %d' % df.shape[0]
#     print 'Total Dementia: %d' % df[df.dementia == 1].shape[0]
#     print 'Total Healthy: %d' % df[df.dementia == 0].shape[0]

# if __name__ == '__main__':
#     blogplot(roc_auc_score)
#     # blogplot(f1_score)

#     # print 'Running: accuracy_score'
#     # blogplot(accuracy_score)
#     # print 'Running: precision_score'
#     # blogplot(precision_score)
#     # print 'Running: recall_score'
#     # blogplot(recall_score)

#     # blogscatter("getSUBTLWordScores", save=True)
#     # blogscatter("MeanWordLength")
#     # blogscatter("NP_to_PRP")
