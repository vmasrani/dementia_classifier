import os

# Data paths
SOUNDFILE_CONTROL_DATA_PATH  = 'dementia_classifier/data/soundfiles/control/'
SOUNDFILE_DEMENTIA_DATA_PATH = 'dementia_classifier/data/soundfiles/dementia/'

DISCOURSE_CONTROL_DATA_PATH  = 'dementia_classifier/data/discourse_trees/control/'
DISCOURSE_DEMENTIA_DATA_PATH = 'dementia_classifier/data/discourse_trees/dementia/'

DBANK_DATA_PATH   = 'dementia_classifier/data/dementiabank'
DBANK_PICKLE_PATH = "dementia_classifier/data/dementiabank.pkl"

DBANK_AGE_GENDER = 'dementia_classifier/data/dementiabank_info/age_gender.txt'
DBANK_DIAGNOSIS  = 'dementia_classifier/data/dementiabank_info/diagnosis.txt'

BLOG_CORPUS_PATH = 'dementia_classifier/data/blog_corpus.xml'
BLOG_PICKLE_PATH = 'dementia_classifier/data/blog_corpus.pkl'
BLOG_FILTER_PATH = 'dementia_classifier/data/blog_filters/'

SCA_FOLDER   = 'dementia_classifier/lib/SCA/L2SCA/'
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
PLOT_PATH    = os.path.join(PROJECT_PATH, 'plots/')
# =================================================================
# ----------------------# SQL Table name---------------------------
# =================================================================

# Dementiabank Data
SQL_DBANK_TEXT_FEATURES = "dementiabank_text_features"
SQL_DBANK_DIAGNOSIS = "dementiabank_diagnosis"
SQL_DBANK_DEMOGRAPHIC = "dementiabank_demographic"
SQL_DBANK_ACOUSTIC_FEATURES = "dementiabank_acoustic_features"
SQL_DBANK_DISCOURSE_FEATURES = "dementiabank_discourse_features"

# Blogs
SQL_BLOG_SUFFIX = 'text_features'
SQL_BLOG_QUALITY = 'blog_quality'

# Results 
DOMAIN_ADAPTATION_RESULTS_PREFIX = 'results_domain_adaptation'
ABLATION_RESULTS_PREFIX = 'results_ablation'
NEW_FEATURES_RESULTS_PREFIX = 'results_new_features'
BLOG_RESULTS = 'results_blog'
BLOG_ABLATION_PREFIX = 'results_blog_ablation'

PARSER_MAX_LENGTH = 50


