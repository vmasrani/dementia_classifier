import numpy as np
import pandas as pd
from scipy import stats
from dementia_classifier.analysis import util
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from scipy.cluster.vq import whiten
from sklearn.dummy import DummyClassifier


ALZHEIMERS     = ['PossibleAD', 'ProbableAD']
CONTROL        = ['Control']
NON_ALZHEIMERS = ['MCI', 'Memory', 'Other', 'Vascular']
MCI = ['MCI']


class DementiaCV(object):
    """
    DementiaCV performs 10-fold group cross validation, where data points with a given label
    are confined in a single fold. This object is written with the intention of being a 
    superclass to the DomainAdaptation and BlogAdaptation subclasses, where the subclasses 
    only need to override the 'get_data_folds' method
    """

    def __init__(self, model, X=None, y=None, labels=None, silent=False):
        super(DementiaCV, self).__init__()
        self.model = model

        self.X = X
        self.y = y
        self.labels = labels
        self.columns = X.columns

        self.methods = ['default']
        self.nfolds = 10

        # Results
        self.silent = silent
        self.results    = {}
        self.best_score = {}
        self.best_k     = {}

        self.myprint("Model %s" % model)
        self.myprint("===========================")

    def get_data_folds(self, fold_type='default'):
        X, y, labels = self.X, self.y, self.labels
        if X is None or y is None:
            raise ValueError("X or y is None")

        group_kfold = GroupKFold(n_splits=self.nfolds).split(X, y, groups=labels)
        data = []
        for train_index, test_index in group_kfold:
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)
        return data

    def myprint(self, msg):
        if not self.silent:
            print msg

    def train_model(self, method='default', k_range=None, model=None):
        if model is None:
            model = self.model

        acc = []
        fms = []
        roc = []

        for idx, fold in enumerate(self.get_data_folds(method)):
            self.myprint("Processing fold: %i" % idx)

            X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
            X_test, y_test   = fold["X_test"], fold["y_test"].ravel()

            acc_scores = []
            fms_scores = []
            if y_test.all():
                print "All values in y_test are the same in fold %i, ROC not defined." % idx
            roc_scores = []

            nfeats = X_train.shape[1]
            feats = util.get_top_pearson_features(X_train, y_train, nfeats)
            if k_range is None:
                k_range = range(1, nfeats)
            if k_range[0] == 0:
                raise ValueError("k_range cannot start with 0")
            for k in k_range:
                indices = feats[:k]
                # Select k features
                X_train_fs = X_train[:, indices]
                X_test_fs  = X_test[:, indices]

                model = model.fit(X_train_fs, y_train)

                # Predict
                yhat_probs = model.predict_proba(X_test_fs)
                yhat = model.predict(X_test_fs)

                # Save
                acc_scores.append(accuracy_score(y_test, yhat))
                fms_scores.append(f1_score(y_test, yhat))
                if y_test.all():
                    roc_scores.append(np.nan)
                else:
                    roc_scores.append(roc_auc_score(y_test, yhat_probs[:, 1]))
            # ----- save fold -----
            acc.append(acc_scores)
            fms.append(fms_scores)
            roc.append(roc_scores)

        self.results[method] = {"acc": np.asarray(acc),
                                "fms": np.asarray(fms),
                                "roc": np.asarray(roc)
                                }

        self.best_k[method]  = {"acc": np.array(k_range)[np.argmax(np.nanmean(acc, axis=0))],
                                "fms": np.array(k_range)[np.argmax(np.nanmean(fms, axis=0))],
                                "roc": np.array(k_range)[np.argmax(np.nanmean(roc, axis=0))],
                                "k_range": k_range}

        self.best_score[method] = {"acc": np.max(np.nanmean(acc, axis=0)),
                                   "fms": np.max(np.nanmean(fms, axis=0)),
                                   "roc": np.max(np.nanmean(roc, axis=0))
                                   }

        return self

    def feature_rank(self, method='default', thresh=50):
        nfeats = self.columns.shape[0]
        nfolds = self.nfolds
        feat_scores = pd.DataFrame(np.zeros([nfeats, nfolds]), columns=range(nfolds), index=self.columns)

        for fold_idx, fold in enumerate(self.get_data_folds(method)):
            X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
            ranked_features = util.get_top_pearson_features(X_train, y_train, nfeats)
            for rank, feat_idx in enumerate(ranked_features[:thresh]):
                feature = self.columns[feat_idx]
                weight = (thresh - rank) / float(thresh)
                feat_scores[fold_idx].ix[feature] = weight

        # Drop rows with all zeros
        df = feat_scores[(feat_scores.T != 0).any()]
        df = df.stack().reset_index()
        df.columns = ['feature', 'fold', 'weight']
        
        return df


class DomainAdaptationCV(DementiaCV):
    """Subclass of DementiaCV where the six domain adaptation methods 
    are computed.
    """

    def __init__(self, model, Xt, yt, Xs, ys, silent=False, random_state=1):
        super(DomainAdaptationCV, self).__init__(model, X=Xt, y=yt, silent=silent)
        self.silent = silent
        self.Xt, self.yt = Xt, yt  # Save target data + labels
        self.Xs, self.ys = Xs, ys  # Save source data + labels
        self.methods = ['target_only', 'source_only', 'relabeled', 'augment', 'coral']

    def get_data_folds(self, fold_type):
        if fold_type not in self.methods:
            raise KeyError('fold_type not one of: %s' % self.methods)

        data = []
        lt = self.Xt.index.values
        ls = self.Xs.index.values
        Xs, ys = self.Xs.values, self.ys.values
        Xt, yt = self.Xt.values, self.yt.values

        group_kfold = GroupKFold(n_splits=10).split(Xt, yt, groups=lt)
        for train_index, test_index in group_kfold:
            if fold_type == "target_only":
                X_train = Xt[train_index]
                y_train = yt[train_index]
                X_test  = Xt[test_index]
                y_test  = yt[test_index]
                train_labels = np.array(lt)[train_index]
            elif fold_type == 'source_only':
                X_train = Xs
                y_train = ys
                X_test  = Xt[test_index]
                y_test  = yt[test_index]
                train_labels = ls
            elif fold_type == 'relabeled':
                # merge'em
                X_merged_relab = np.concatenate([Xs, Xt[train_index]])
                y_merged_relab = np.concatenate([ys, yt[train_index]])
                train_labels   = np.concatenate([np.array(ls), np.array(lt)[train_index]])
                # shuffle
                X_train_relab, y_train_relab, train_labels = shuffle(
                    X_merged_relab, y_merged_relab, train_labels, random_state=1)
                X_train = X_train_relab
                y_train = y_train_relab
                X_test  = Xt[test_index]
                y_test  = yt[test_index]
            elif fold_type == 'augment':
                # Extend feature space (train)
                X_merged_aug = self.merge_and_extend_feature_space(Xt[train_index], Xs)
                y_merged_aug = np.concatenate([yt[train_index], ys])
                train_labels = np.concatenate([np.array(lt)[train_index], np.array(ls)])
                # Extend feature space (test)
                X_test_aug = self.merge_and_extend_feature_space(Xt[test_index])
                X_train = X_merged_aug
                y_train = y_merged_aug
                X_test  = X_test_aug
                y_test  = yt[test_index]
            elif fold_type == 'coral':
                # ---------coral------------
                X_train = self.CORAL(Xs, Xt[train_index])
                y_train = ys
                X_test  = Xt[test_index]
                y_test  = yt[test_index]
                train_labels = ls
            else:
                raise KeyError('fold_type not one of: %s' % self.models)
            fold = {}
            fold["X_train"] = X_train
            fold["y_train"] = y_train
            fold["X_test"]  = X_test
            fold["y_test"]  = y_test
            fold["train_labels"] = train_labels
            data.append(fold)
        return data

    def train_all(self, k_range=None):
        for method in self.methods:
            self.myprint("\nTraining: %s" % method)
            self.myprint("---------------------------")
            self.train_model(method, k_range=k_range)

    def train_majority_class(self):
        self.myprint("\nTraining Majority Class")
        self.myprint("===========================")
        lt     = self.Xt.index.values
        Xt, yt = self.Xt.values, self.yt.values
        group_kfold = GroupKFold(n_splits=10).split(Xt, yt, groups=lt)

        acc_scores = []
        fms_scores = []

        for train_index, test_index in group_kfold:
            # Data is same as target_only data
            y_train, y_test = yt[train_index], yt[test_index]
            labels          = np.array(lt)[train_index]
            patient_ids     = np.unique(labels)
            maj = []

            # Need to predict most common patient type, not most common interview type
            for patient in patient_ids:
                ind = np.where(labels == patient)[0]
                patient_type = y_train[ind].flatten()[0]
                maj.append(patient_type)

            maj = stats.mode(maj)[0]
            yhat = np.full(y_test.shape, maj, dtype=bool)

            acc_scores.append(accuracy_score(y_test, yhat))     # Save
            fms_scores.append(f1_score(y_test, yhat))           # Save

        # ----- save row -----
        self.results['majority_class'] = {"acc": np.asarray(acc_scores), "fms": np.asarray(fms_scores)}

    # map row to new feature space
    # in accordance with 'frustratingly simple' paper
    def merge_and_extend_feature_space(self, X_target, X_source=None):
        X_target_extended = np.concatenate([X_target, np.zeros(X_target.shape), X_target], axis=1)
        if X_source is None:
            return X_target_extended
        else:
            X_source_extended = np.concatenate([X_source, X_source, np.zeros(X_source.shape)], axis=1)
            return np.concatenate([X_target_extended, X_source_extended])

        # Following CORAL paper -http://arxiv.org/abs/1511.05547

    # Algorithm 1
    def CORAL(self, Ds, Dt):
        EPS = 1
        N, D = Ds.shape
        # # Normalize (Here 'whiten' divides by std)
        Ds = whiten(Ds - Ds.mean(axis=0))
        Dt = whiten(Dt - Dt.mean(axis=0))

        Cs = np.cov(Ds, rowvar=False) + EPS * np.eye(D)
        Ct = np.cov(Dt, rowvar=False) + EPS * np.eye(D)

        Ws = util.whiten_matrix(Cs)
        Wcolor = util.color_matrix(Ct)

        Ds = np.dot(Ds, Ws)      # Whiten
        Ds = np.dot(Ds, Wcolor)  # Recolor

        assert not np.isnan(Ds).any()
        return Ds


class BlogCV(DementiaCV):
    """BlogCV is a subclass of DementiaCV which performs a 9-fold cross validation 
    where the test fold has contains posts from blogs not in the training fold.
    """

    def __init__(self, model, X, y, labels, silent=False, random_state=1):
        super(BlogCV, self).__init__(model, X=X, y=y, labels=labels, silent=silent)
        self.methods = ['model', 'majority_class']
        
    def get_data_folds(self, fold_type='default'):

        X, y, labels = self.X, self.y, self.labels

        testset1 = ["creatingmemories", "journeywithdementia"]
        testset2 = ["creatingmemories", "earlyonset"]
        testset3 = ["creatingmemories", "helpparentsagewell"]

        testset4 = ["living-with-alzhiemers", "journeywithdementia"]
        testset5 = ["living-with-alzhiemers", "earlyonset"]
        testset6 = ["living-with-alzhiemers", "helpparentsagewell"]

        testset7 = ["parkblog-silverfox", "journeywithdementia"]
        testset8 = ["parkblog-silverfox", "earlyonset"]
        testset9 = ["parkblog-silverfox", "helpparentsagewell"]

        folds = [testset1, testset2, testset3, testset4, testset5, testset6, testset7, testset8, testset9]
        data = []
        for fold in folds:
            train_index = ~X.index.isin(fold)
            test_index = X.index.isin(fold)
            fold = {}
            fold["X_train"] = X.values[train_index]
            fold["y_train"] = y.values[train_index]
            fold["X_test"]  = X.values[test_index]
            fold["y_test"]  = y.values[test_index]
            fold["train_labels"]  = np.array(labels)[train_index]
            data.append(fold)

        return data
