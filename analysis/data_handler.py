from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from dementia_classifier.feature_extraction.feature_sets import feature_set_list

# --------MySql---------
from dementia_classifier import db
cnx = db.get_connection()
# ----------------------

ALZHEIMERS     = ["PossibleAD", "ProbableAD"]
CONTROL        = ["Control"]
MCI            = ["MCI"]
NON_ALZHEIMERS = ["MCI", "Memory", "Other", "Vascular"]
CONTROL_BLOGS  = ["earlyonset", "helpparentsagewell", "journeywithdementia"]
DEMENTIA_BLOGS = ["creatingmemories", "living-with-alzhiemers", "parkblog-silverfox"]

# ------------------
# Diagnosis keys
# - Control
# - MCI
# - Memory
# - Other
# - PossibleAD
# - ProbableAD
# - Vascular
# ------------------

# ===================================================================================
# ----------------------------------DementiaBank-------------------------------------
# ===================================================================================


def get_data(diagnosis=ALZHEIMERS + CONTROL, drop_features=None, polynomial_terms=None):

    # Read from sql
    text = pd.read_sql_table("dementiabank_text_features", cnx)
    demo = pd.read_sql_table("dementiabank_demographic", cnx)
    diag = pd.read_sql_table("dementiabank_diagnosis", cnx)
    disc = pd.read_sql_table("dementiabank_discourse_features", cnx)
    acoustic = pd.read_sql_table("dementiabank_acoustic_features", cnx)

    # Add diagnosis
    diag = diag[diag['diagnosis'].isin(diagnosis)]
    fv = pd.merge(text, diag)
    # Merge lexical and acoustic
    fv = pd.merge(fv, acoustic, on=['interview'])
    # Add demographics
    fv = pd.merge(fv, demo)
    # Add discourse
    fv = pd.merge(fv, disc, on=['interview'])

    # Randomize
    fv = fv.sample(frac=1, random_state=20)

    # Collect Labels
    labels = [label[:3] for label in fv['interview']]
    # Diagnoses not in control marked receive positive label
    y = ~fv.diagnosis.isin(CONTROL)
    # Clean
    drop = ['level_0', 'interview', 'diagnosis', 'gender', 'index', 'gender_int']

    X = fv.drop(drop, axis=1, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')

    X.index = labels
    y.index = labels

    if drop_features:
        X = X.drop(drop_features, axis=1, errors='ignore')

    X = make_polynomial_terms(X, polynomial_terms)

    return X, y, labels


def make_polynomial_terms(data, cols):
    if cols is None:
        return data

    for f1, f2 in itertools.combinations_with_replacement(cols, 2):
        if f1 == f2:
            prefix = 'sqr_'
        else:
            prefix = 'intr_'
        data[prefix + f1 + "_" + f2] = data[f1] * data[f2]

    data = data.drop(cols, axis=1, errors='ignore')

    return data


def get_target_source_data(random_state=1):
    # Get data
    feature_set = feature_set_list.new_features()

    X_alz, y_alz, l_alz = get_data(diagnosis=ALZHEIMERS, drop_features=feature_set)
    X_con, y_con, l_con = get_data(diagnosis=CONTROL, drop_features=feature_set)
    X_mci, y_mci, l_mci = get_data(diagnosis=MCI, drop_features=feature_set)

    # Split control samples into target/source set (making sure one patient doesn't appear in both t and s)
    gkf = GroupKFold(n_splits=6).split(X_con, y_con, groups=l_con)
    source, target = gkf.next()
    Xt, yt = concat_and_shuffle(X_mci, y_mci, l_mci, X_con.ix[target], y_con.ix[target], np.array(l_con)[target], random_state=random_state)
    Xs, ys = concat_and_shuffle(X_alz, y_alz, l_alz, X_con.ix[source], y_con.ix[source], np.array(l_con)[source], random_state=random_state)

    return Xt, yt, Xs, ys


def concat_and_shuffle(X1, y1, l1, X2, y2, l2, random_state=1):
    pd.options.mode.chained_assignment = None  # default='warn'
    # Coerce all arguments to dataframes
    X1, X2 = pd.DataFrame(X1), pd.DataFrame(X2)
    y1, y2 = pd.DataFrame(y1), pd.DataFrame(y2)
    l1, l2 = pd.DataFrame(l1), pd.DataFrame(l2)

    X_concat = X1.append(X2, ignore_index=True)
    y_concat = y1.append(y2, ignore_index=True)
    l_concat = l1.append(l2, ignore_index=True)

    X_shuf, y_shuf, l_shuf = shuffle(X_concat, y_concat, l_concat, random_state=random_state)

    X_shuf['labels'] = l_shuf
    y_shuf['labels'] = l_shuf

    X_shuf.set_index('labels', inplace=True)
    y_shuf.set_index('labels', inplace=True)

    return X_shuf, y_shuf

# ===================================================================================
# ----------------------------------BlogData-----------------------------------------
# ===================================================================================


def get_blog_data(keep_only_good=True, random=20, drop_features=None):
    # Read from sql
    cutoff_date = pd.datetime(2017, 4, 4)  # April 4th 2017 was when data was collected for ACL paper

    demblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in DEMENTIA_BLOGS])
    ctlblogs = pd.concat([pd.read_sql_table("%s_text_features" % blog, cnx) for blog in CONTROL_BLOGS])
    qual     = pd.read_sql_table("blog_quality", cnx)
    
    demblogs['dementia'] = True
    ctlblogs['dementia'] = False
    
    fv = pd.concat([demblogs, ctlblogs], ignore_index=True)

    # Remove recent posts (e.g. after paper was published)
    qual['date'] = pd.to_datetime(qual.date)
    qual = qual[qual.date < cutoff_date] 

    # keep only good blog posts 
    if keep_only_good:
        qual = qual[qual.quality == 'good']

    demblogs = pd.merge(demblogs, qual[['id', 'blog']], on=['id', 'blog'])

    # Randomize
    fv = fv.sample(frac=1, random_state=random)

    # Get labels
    labels = fv['blog'].tolist()

    # Split
    y = fv['dementia'].astype('bool')

    # Clean
    drop = ['blog', 'dementia', 'id']
    X = fv.drop(drop, 1, errors='ignore')

    if drop_features:
        X = X.drop(drop_features, axis=1, errors='ignore')


    X = X.apply(pd.to_numeric, errors='ignore')
    X.index = labels
    y.index = labels

    return X, y, labels


def get_blog_scatterplot_data(keep_only_good=True, random=20):
    # Read from sql
    blogs = pd.read_sql_table("blogs", cnx)
    qual = pd.read_sql_table("blog_quality", cnx)
    lengths = pd.read_sql_table("blog_lengths", cnx)

    if keep_only_good:
        qual = qual[qual.quality == 'good']

    # keep only good
    data = pd.merge(blogs, qual[['id', 'blog', 'date']], on=['id', 'blog'])
    data = pd.merge(data, lengths, on=['id', 'blog'])

    # Fix reverse post issue
    data.id = data.id.str.split('_', expand=True)[1].astype(int)
    data.id = -data.id

    data.date = pd.to_datetime(data.date)

    return data


# =================================================================
# ------------------------ Read Results----------------------------
# =================================================================
def get_max_fold(df):
    max_k = df.mean().argmax()
    df = df[max_k].to_frame()
    df.columns = ['folds']
    df['max_k'] = int(max_k)
    return df.reset_index(drop=True)


# Returns 10 folds for best k (k= number of features)
def get_da_results(classifier_name, domain_adapt_method, metric):
    name = "results_domain_adaptation_%s_10fold" % classifier_name 
    table = pd.read_sql_table(name, cnx, index_col='index')
    table = table[(table.metric == metric) & (table.method == domain_adapt_method)].dropna(axis=1)
    df = get_max_fold(table)
    df['model'] = classifier_name
    df['method'] = domain_adapt_method
    return df


def get_ablation_results(ablation_set, model, metric):
    reference = "results_ablation_none"

    ref = pd.read_sql_table(reference, cnx, index_col='index')
    ref = ref[(ref.metric == metric) & (ref.model == model)].dropna(axis=1)
    
    max_ref_k = ref.mean().argmax()
    ref = ref[max_ref_k].to_frame().reset_index(drop=True)
    ref.columns = ['folds']

    # abl == ablated 
    name = "results_ablation_%s" % ablation_set 
    abl = pd.read_sql_table(name, cnx, index_col='index')
    abl = abl[(abl.metric == metric) & (abl.model == model)].dropna(axis=1)

    max_abl_k = abl.mean().argmax()
    abl = abl[max_abl_k].to_frame().reset_index(drop=True)
    abl.columns = ['folds']

    # Difference from reference
    diff = abl['folds'].mean() - ref['folds'].mean()
    diff = pd.DataFrame([diff * 100.0], columns=['folds'])
    diff['model'] = model
    diff['metric'] = metric
    diff['ablation_set'] = ablation_set
    
    return diff


def get_new_feature_results(new_feature_set, model, metric):
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

    # Difference from reference
    diff = nfs['folds'].mean() - ref['folds'].mean()
    diff = pd.DataFrame([diff * 100.0], columns=['folds'])
    diff['model'] = model
    diff['metric'] = metric
    diff['new_feature_set'] = new_feature_set
    
    return diff


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

