from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

CLASSIFIERS = {
    "DummyClassifier": DummyClassifier("most_frequent"),
    "LogReg": LogisticRegression(),
    "KNeighbors": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(max_depth=3, n_jobs=1, n_estimators=100),
    "GausNaiveBayes": GaussianNB(),
    "SVC": SVC(probability=True),
}


CLASSIFIER_SET_1 = [
    "DummyClassifier",
    "LogReg",
    "SVC",
]

CLASSIFIER_SET_2 = [
    "DummyClassifier",
    "RandomForest",
    "KNeighbors",
    "GausNaiveBayes"
]

# Ordered
CLASSIFIER_KEYS = [
    "DummyClassifier",
    "LogReg",
    "SVC",
    "KNeighbors",
    "RandomForest",
    # "MLP",
    "GausNaiveBayes"]

# Feature set
FEATURE_SETS = [
    "cfg",
    "syntactic_complexity",
    "psycholinguistic",
    "vocabulary_richness",
    "repetitiveness",
    "acoustics",
    "demographic",
    "parts_of_speech",
    "information_content",
]

NEW_FEATURE_SETS = [
    "strips",
    "halves",
    "quadrant",
    "discourse",
]

BLOG_FEATURE_SETS = [
    "cfg",
    "syntactic_complexity",
    "psycholinguistic",
    "vocabulary_richness",
    "repetitiveness",
    "parts_of_speech",
]

BLOG_NAMES = [
    "journeywithdementia",
    "helpparentsagewell",
    "earlyonset",
    "living-with-alzhiemers",
    "parkblog-silverfox",
    "creatingmemories",
]

METRICS = ['acc', 'roc', 'fms']