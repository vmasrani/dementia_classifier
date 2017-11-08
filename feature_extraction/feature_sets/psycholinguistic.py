import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import wordnet as wn


# Global psycholinguistic data structures
FEATURE_DATA_PATH = 'dementia_classifier/feature_extraction/feature_sets/psycholing_scores/'

# Made global so files only need to be read once
psycholinguistic_scores = {}
SUBTL_cached_scores = {}

# -----------Global Tools--------------
# remove punctuation, lowercase, stem
stemmer = nltk.PorterStemmer()
stop = stopwords.words('english')


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]
# -------------------------------------

# Constants
LIGHT_VERBS       = ["be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put"]
VERB_POS_TAGS     = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
NOUN_POS_TAGS     = ["NN", "NNS", "NNP", "NNPS", ]
FEATURE_DATA_LIST = ["familiarity", "concreteness", "imagability", 'aoa']

# # Information Unit Words (old)
# BOY       = ['boy', 'son', 'brother', 'male child']
# GIRL      = ['girl', 'daughter', 'sister', 'female child']
# WOMAN     = ['woman', 'mom', 'mother', 'lady', 'parent']
# KITCHEN   = ['kitchen']
# EXTERIOR  = ['exterior', 'outside', 'garden', 'yard']
# COOKIE    = ['cookie']
# JAR       = ['jar']
# STOOL     = ['stool']
# SINK      = ['sink']
# PLATE     = ['plate']
# DISHCLOTH = ['dishcloth', 'rag', 'cloth', 'napkin', 'towel']
# WATER     = ['water']
# WINDOW    = ['window']
# CUPBOARD  = ['cupboard']
# DISHES    = ['dishes']
# CURTAINS  = ['curtains', 'curtain']
# # Action words
# STEAL    = ['take', 'steal', 'taking', 'stealing']
# FALL     = ['fall', 'falling', 'slip', 'slipping']
# WASH     = ['wash', 'dry', 'clean', 'washing', 'drying', 'cleaning']
# OVERFLOW = ['overflow', 'spill', 'overflowing', 'spilling']

# Extended keyword set
BOY       = ['boy', 'son', 'brother', 'male child']
GIRL      = ['girl', 'daughter', 'sister', 'female child']
WOMAN     = ['woman', 'mom', 'mother', 'lady', 'parent', 'female', 'adult', 'grownup']
KITCHEN   = ['kitchen', 'room']
EXTERIOR  = ['exterior', 'outside', 'garden', 'yard', 'outdoors', 'backyard', 'driveway', 'path', 'tree', 'bush']
COOKIE    = ['cookie', 'biscuit', 'cake', 'treat']
JAR       = ['jar', 'container', 'crock', 'pot']
STOOL     = ['stool', 'seat', 'chair', 'ladder']
SINK      = ['sink', 'basin', 'washbasin', 'washbowl', 'washstand', 'tap']
PLATE     = ['plate']
DISHCLOTH = ['dishcloth', 'dishrag', 'rag', 'cloth', 'napkin', 'towel']
WATER     = ['water', 'dishwater', 'liquid']
WINDOW    = ['window', 'frame', 'glass']
CUPBOARD  = ['cupboard', 'closet', 'shelf']
DISHES    = ['dish', 'dishes', 'cup', 'cups', 'counter']
CURTAINS  = ['curtain', 'curtains', 'drape', 'drapes', 'drapery', 'drapery', 'blind', 'blinds', 'screen', 'screens']

# Action words
STEAL    = ['take', 'steal', 'taking', 'stealing']
FALL     = ['fall', 'falling', 'slip', 'slipping']
WASH     = ['wash', 'dry', 'clean', 'washing', 'drying', 'cleaning']
OVERFLOW = ['overflow', 'spill', 'overflowing', 'spilling']


# ================================================
# -------------------Tools------------------------
# ================================================


def getAllWordsFromInterview(interview):
    words = []
    for uttr in interview:
        words += [word.lower() for word in uttr["token"] if word.isalpha()]
    return words


def getAllNonStopWordsFromInterview(interview):
    words = []
    for uttr in interview:
        words += [word.lower() for word in uttr["token"] if word.isalpha() and word not in stop]
    return words

# ================================================
# -----------Psycholinguistic features------------
# ================================================

# Input: one of "familiarity", "concreteness", "imagability", or 'aoa'
# Output: none
# Notes: Makes dict mapping words to score, store dict in psycholinguistic


def _load_scores(name):
    if name not in FEATURE_DATA_LIST:
        raise ValueError("name must be one of: " + str(FEATURE_DATA_LIST))
    with open(FEATURE_DATA_PATH + name) as file:
        d = {word.lower(): float(score) for (score, word) in [line.strip().split(" ") for line in file]}
        psycholinguistic_scores[name] = d

# Input: Interview is a list of utterance dictionaries, measure is one of "familiarity", "concreteness", "imagability", or 'aoa'
# Output: PsycholinguisticScore for a given measure


def getPsycholinguisticScore(interview, measure):
    if measure not in FEATURE_DATA_LIST:
        raise ValueError("name must be one of: " + str(FEATURE_DATA_LIST))
    if measure not in psycholinguistic_scores:
        _load_scores(measure)
    score = 0
    validwords = 1
    allwords = getAllNonStopWordsFromInterview(interview)
    for w in allwords:
        if w.lower() in psycholinguistic_scores[measure]:
            score += psycholinguistic_scores[measure][w.lower()]
            validwords += 1
    # Only normalize by words present in dict
    return score / validwords

# Input: list of words
# Output: scores for each word
# Notes: This gets the SUBTL frequency count for a word from http://subtlexus.lexique.org/moteur2/index.php


def _getSUBTLWordScoresFromURL(wordlist):
    # Use set so words are unique
    unknown_words = set([w.lower() for w in wordlist if w not in SUBTL_cached_scores])
    # Load into cache all unknown words
    if unknown_words:
        url = 'http://subtlexus.lexique.org/moteur2/simple.php'
        encoded_words = '\n'.join(unknown_words)
        params = {'database': 'subtlexus', 'mots': encoded_words}
        r = requests.get(url, params=params)
        rows = []
        table = BeautifulSoup(r.content, "html.parser")
        # Parse datatable to get SUBTLwf scores
        for row in table.findAll("tr"):
            cells = row.findAll("td")
            row = [c.findAll(text=True)[0] for c in cells[:10]]
            rows.append(row)
        # Fill dictionary, ignore header row
        for row in rows[1:]:
            SUBTL_cached_scores[row[0]] = float(row[5])
            unknown_words.remove(row[0])
        # Words remaining in unknown words don't have SUBTL word scores.
        # Add them to dict with score of 0 to prevent redundant calls to
        for word in unknown_words:
            SUBTL_cached_scores[word] = 0

    # Read the scores for each word
    # (Ignores words which don't have score)
    return [SUBTL_cached_scores[w] for w in wordlist if w in SUBTL_cached_scores]


def getSUBTLWordScores(interview):
    allwords = getAllNonStopWordsFromInterview(interview)
    scores = _getSUBTLWordScoresFromURL(allwords)
    return 0 if len(allwords) == 0 else sum(scores) / len(allwords)

# Input: Interview is a list of utterance
# Output: Normalized count of light verbs


def getLightVerbCount(interview):
    light_verbs = 0.0
    total_verbs = 0.0
    for uttr in interview:
        for w in uttr['pos']:
            if w[0].lower() in LIGHT_VERBS:
                light_verbs += 1
            if w[1] in VERB_POS_TAGS:
                total_verbs += 1
    return 0 if total_verbs == 0 else light_verbs / total_verbs

# ================================================
# -----------Information Unit features------------
# Taken from http://www.sciencedirect.com/science/article/pii/S0093934X96900334
# End of page 4
# ================================================
# Input: Sent is a sentence dictionary,
# Output: Binary value (1/0) of sentence contains info unit
# Notes: Info units are hard coded keywords (Need to find paper or something to justify hard coded words)


# --------------
# Subjects (3)
# -------------

def keywordIUSubjectBoy(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in BOY]
    return len(keywords)


def binaryIUSubjectBoy(interview):
    count = keywordIUSubjectBoy(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUSubjectGirl(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in GIRL]
    return len(keywords)


def binaryIUSubjectGirl(interview):
    count = keywordIUSubjectGirl(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUSubjectWoman(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in WOMAN]
    return len(keywords)


def binaryIUSubjectWoman(interview):
    count = keywordIUSubjectWoman(interview)
    return 0 if count == 0 else 1


# --------------
# Places (2)
# -------------


def keywordIUPlaceKitchen(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in KITCHEN]
    return len(keywords)


def binaryIUPlaceKitchen(interview):
    count = keywordIUPlaceKitchen(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUPlaceExterior(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in EXTERIOR]
    return len(keywords)


def binaryIUPlaceExterior(interview):
    count = keywordIUPlaceExterior(interview)
    return 0 if count == 0 else 1


# --------------
# Objects (11)
# -------------


def keywordIUObjectCookie(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in COOKIE]
    return len(keywords)


def binaryIUObjectCookie(interview):
    count = keywordIUObjectCookie(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectJar(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in JAR]
    return len(keywords)


def binaryIUObjectJar(interview):
    count = keywordIUObjectJar(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectStool(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in STOOL]
    return len(keywords)


def binaryIUObjectStool(interview):
    count = keywordIUObjectStool(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectSink(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in SINK]
    return len(keywords)


def binaryIUObjectSink(interview):
    count = keywordIUObjectSink(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectPlate(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in PLATE]
    return len(keywords)


def binaryIUObjectPlate(interview):
    count = keywordIUObjectPlate(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectDishcloth(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in DISHCLOTH]
    return len(keywords)


def binaryIUObjectDishcloth(interview):
    count = keywordIUObjectDishcloth(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectWater(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in WATER]
    return len(keywords)


def binaryIUObjectWater(interview):
    count = keywordIUObjectWater(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectWindow(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in WINDOW]
    return len(keywords)


def binaryIUObjectWindow(interview):
    count = keywordIUObjectWindow(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectCupboard(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in CUPBOARD]
    return len(keywords)


def binaryIUObjectCupboard(interview):
    count = keywordIUObjectCupboard(interview)
    return 0 if count == 0 else 1

# -----


def keywordIUObjectDishes(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in DISHES]
    return len(keywords)


def binaryIUObjectDishes(interview):
    count = keywordIUObjectDishes(interview)
    return 0 if count == 0 else 1

#-----


def keywordIUObjectCurtains(interview):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in CURTAINS]
    return len(keywords)


def binaryIUObjectCurtains(interview):
    count = keywordIUObjectCurtains(interview)
    return 0 if count == 0 else 1


#-------------
# Actions (7)
#-------------

# For action unit to be present, the subject and action (eg. 'boy' and 'fall')
# must be tagged together in the utterance
# Input: POSTags, subject list, verb list
def check_action_unit(pos_tags, subjs, verbs):
    stemmed_subjs = [stemmer.stem(s) for s in subjs]
    stemmed_verbs = [stemmer.stem(s) for s in verbs]
    subj_found, verb_found = False, False
    for pos in pos_tags:
        if stemmer.stem(pos[0]) in stemmed_subjs and pos[1] in NOUN_POS_TAGS:
            subj_found = True
        if stemmer.stem(pos[0]) in stemmed_verbs and pos[1] in VERB_POS_TAGS:
            verb_found = True
    return subj_found and verb_found


# # boy taking or stealing
# def binaryIUActionBoyTaking(interview):
#     for uttr in interview:
#         if(check_action_unit(uttr['pos'], BOY, ['take', 'steal'])):
#             return 1
#     return 0

# # boy or stool falling


# def binaryIUActionStoolFalling(interview):
#     for uttr in interview:
#         if(check_action_unit(uttr['pos'], BOY + ['stool'], ['falling'])):
#             return 1
#     return 0

# # Woman drying or washing dishes/plate


# def binaryIUActionWomanDryingWashing(interview):
#     for uttr in interview:
#         if(check_action_unit(uttr['pos'], WOMAN + ['dish', 'plate'], ['wash', 'dry'])):
#             return 1
#     return 0

# # Water overflowing or spilling


# def binaryIUActionWaterOverflowing(interview):
#     for uttr in interview:
#         if(check_action_unit(uttr['pos'], ['water', 'tap', 'sink'], ['overflow', 'spill'])):
#             return 1
#     return 0


# For action unit to be present, the subject and action (eg. 'boy' and 'fall')
# must be tagged together in the utterance
# Input: POSTags, subject list, verb list

# boy taking or stealing
def keywordIUActionBoyTaking(interview):
    count = 0
    for uttr in interview:
        if(check_action_unit(uttr['pos'], BOY, STEAL)):
            count += 1
    return count


def binaryIUActionBoyTaking(interview):
    count = keywordIUActionBoyTaking(interview)
    return 0 if count == 0 else 1

# boy or stool falling


def keywordIUActionStoolFalling(interview):
    count = 0
    for uttr in interview:
        if(check_action_unit(uttr['pos'], BOY + STOOL, FALL)):
            count += 1
    return count


def binaryIUActionStoolFalling(interview):
    count = keywordIUActionStoolFalling(interview)
    return 0 if count == 0 else 1


# Woman drying or washing dishes/plate
def keywordIUActionWomanDryingWashing(interview):
    count = 0
    for uttr in interview:
        if(check_action_unit(uttr['pos'], WOMAN + PLATE, WASH)):
            count += 1
    return count


def binaryIUActionWomanDryingWashing(interview):
    count = keywordIUActionWomanDryingWashing(interview)
    return 0 if count == 0 else 1

# Water overflowing or spilling


def keywordIUActionWaterOverflowing(interview):
    count = 0
    for uttr in interview:
        if(check_action_unit(uttr['pos'], WATER, OVERFLOW)):
            count += 1
    return count


def binaryIUActionWaterOverflowing(interview):
    count = keywordIUActionWaterOverflowing(interview)
    return 0 if count == 0 else 1


#-----------------------
# General keywords (7)
#-----------------------

# Raw count keywords
# (proxy for the 'keyword count' features)
def count_of_general_keyword(interview, keyword_set):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in keyword_set]
    if not words or not keywords:
        return 0
    else:
        return len(keywords)


# Keywords / all words uttered
# (this is a measure of how 'relevant' the speech is)
def general_keyword_to_non_keyword_ratio(interview, keyword_set):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in keyword_set]
    if not words or not keywords:
        return 0
    else:
        return len(keywords) / float(len(words))


# unique keywords uttered / total set of possible keywords
# (proxy for the 'binary count' features)
def percentage_of_general_keywords_mentioned(interview, keyword_set):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in keyword_set]
    if not words or not keywords:
        return 0
    else:
        return len(set(keywords)) / float(len(keyword_set))


# unique keywords uttered / total_keywords_uttered
# (Measure of the diversity of keywords uttered)
def general_keyword_type_to_token_ratio(interview, keyword_set):
    words = getAllWordsFromInterview(interview)
    keywords = [w for w in words if w in keyword_set]
    if not words or not keywords:
        return 0
    else:
        return len(set(keywords)) / float(len(keywords))


#=====================================
# Feature sets
#=====================================
def get_keyword_set():
    return BOY + GIRL + WOMAN + KITCHEN + EXTERIOR + COOKIE + JAR + STOOL + SINK + PLATE + DISHCLOTH + WATER + WINDOW + CUPBOARD + DISHES + CURTAINS + STEAL + FALL + WASH + OVERFLOW

# ----------------------
# Divide image in half
# ----------------------


def get_leftside_keyword_set():
    return BOY + GIRL + COOKIE + JAR + STOOL + CUPBOARD + STEAL + FALL + KITCHEN


def get_rightside_keyword_set():
    return WOMAN + EXTERIOR + SINK + PLATE + DISHCLOTH + WATER + WINDOW + DISHES + CURTAINS + WASH + OVERFLOW + CUPBOARD + KITCHEN
# ----------------------

# ----------------------
# Divide image in 4 vertical strips
# ----------------------


def get_farleft_keyword_set():
    return GIRL + COOKIE + JAR + STOOL + CUPBOARD + STEAL + KITCHEN + CUPBOARD


def get_centerleft_keyword_set():
    return BOY + COOKIE + STOOL + STEAL + FALL + KITCHEN + CUPBOARD 


def get_farright_keyword_set():
    return WOMAN + EXTERIOR + SINK + PLATE + DISHCLOTH + WATER + WINDOW + DISHES + CURTAINS + WASH + OVERFLOW + KITCHEN + CUPBOARD


def get_centerright_keyword_set():
    return EXTERIOR + WINDOW + DISHES + CURTAINS + KITCHEN + CUPBOARD

# ----------------------
# Divide image in 4 quadrants
# ----------------------


def get_NW_keyword_set():
    return GIRL + COOKIE + JAR + CUPBOARD + STEAL + BOY + COOKIE + KITCHEN


def get_NE_keyword_set():
    return WOMAN + EXTERIOR + PLATE + DISHCLOTH + WASH + WINDOW + CURTAINS + KITCHEN


def get_SE_keyword_set():
    return WOMAN + SINK + WATER + DISHES + OVERFLOW + CUPBOARD + KITCHEN


def get_SW_keyword_set():
    return GIRL + STOOL + FALL + CUPBOARD + KITCHEN


# # -------------------------------------
# # LS/RS switch
# # -------------------------------------
def count_ls_rs_switches(interview):
    leftside  = get_leftside_keyword_set()
    rightside = get_rightside_keyword_set()
    words = getAllWordsFromInterview(interview)

    last_side    = None
    current      = None
    switch_count = 0

    for word in words:
        if word in leftside:
            current = 'left'
        if word in rightside:
            current = 'right'

        if last_side is None and current:
            last_side = current
        else:
            if current and last_side and (current != last_side):
                switch_count += 1
                last_side = current

    return switch_count


# -------------------------------------
# Cosine Similarity Between Utterances
# -------------------------------------


def not_only_stopwords(text):
    unstopped = [w for w in normalize(text) if w not in stop]
    return len(unstopped) != 0


def normalize(text):
    text = str(text).lower().translate(None, string.punctuation)
    return stem_tokens(nltk.word_tokenize(text))

# input: two strings
# returns: (float) similarity
# Note: returns zero if one string consists only of stopwords


def cosine_sim(text1, text2):
    if not_only_stopwords(text1) and not_only_stopwords(text2):
        # Tfid raises error if text contain only stopwords. Their stopword set is different
        # than ours so add try/catch block for strange cases
        try:
            vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')  # Punctuation remover
            tfidf = vectorizer.fit_transform([text1, text2])
            return ((tfidf * tfidf.T).A)[0, 1]
        except ValueError, e:
            print "Error:", e
            print 'Returning 0 for cos_sim between: "', text1, '" and: "', text2, '"'
            return 0
    else:
        return 0
# input: list of raw utterances
# returns: list of cosine similarity between all pairs


def compare_all_utterances(uttrs):
    # Start with non-empty set
    similarities = [0]
    for i in range(len(uttrs)):
        for j in range(i + 1, len(uttrs)):
            similarities.append(cosine_sim(uttrs[i]['raw'], uttrs[j]['raw']))
    return similarities

# input: list of raw utterances
# returns: (float)average similarity over all similarities


def avg_cos_dist(uttrs):
    similarities = compare_all_utterances(uttrs)
    return reduce(lambda x, y: x + y, similarities) / len(similarities)

# input: list of raw utterances
# returns:(float) Minimum similarity over all similarities


def min_cos_dist(uttrs):
    return min(compare_all_utterances(uttrs))

# input: list of raw utterances
# returns: (float) proportion of similarities below threshold


def proportion_below_threshold(uttrs, thresh):
    similarities = compare_all_utterances(uttrs)
    valid = [s for s in similarities if s <= thresh]
    return len(valid) / float(len(similarities))

# input: list of interview utterances stored as [ [{},{},{}], [{},{},{}] ]
# returns: list of features for each interview


def get_psycholinguistic_features(interview):
    feat_dict = {}
    feat_dict["getFamiliarityScore"] = getPsycholinguisticScore(interview, 'familiarity')
    feat_dict["getConcretenessScore"] = getPsycholinguisticScore(interview, 'concreteness')
    feat_dict["getImagabilityScore"] = getPsycholinguisticScore(interview, 'imagability')
    feat_dict["getAoaScore"] = getPsycholinguisticScore(interview, 'aoa')
    feat_dict["getSUBTLWordScores"] = getSUBTLWordScores(interview)

    feat_dict["getLightVerbCount"] = getLightVerbCount(interview)
    feat_dict["avg_cos_dist"] = avg_cos_dist(interview)
    feat_dict["min_cos_dist"] = min_cos_dist(interview)
    feat_dict["proportion_below_threshold_0"] = proportion_below_threshold(interview, 0)
    feat_dict["proportion_below_threshold_0.3"] = proportion_below_threshold(interview, 0.3)
    feat_dict["proportion_below_threshold_0.5"] = proportion_below_threshold(interview, 0.5)
    return feat_dict


def get_spatial_features(interview, photo_split):
    divisions = ['halves', 'strips', 'quadrants']

    if photo_split not in divisions:
        raise ValueError("'photo_split' must be one of 'halves', 'quadrants' or 'strips', not: %s" % photo_split)

    feat_dict = {}

    if photo_split == 'halves':
        # leftside_keywords
        # (These are keywords which only appear on the left side of the image)
        leftside_keywords = get_leftside_keyword_set()
        feat_dict["ls_count"] = count_of_general_keyword(interview, leftside_keywords)
        feat_dict["ls_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, leftside_keywords)
        feat_dict["ls_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, leftside_keywords)
        feat_dict["prcnt_ls_uttered"] = percentage_of_general_keywords_mentioned(
            interview, leftside_keywords)

        # rightside_keywords
        # (These are keywords which only appear on the right side of the image)
        rightside_keywords = get_rightside_keyword_set()
        feat_dict["rs_count"] = count_of_general_keyword(interview, rightside_keywords)
        feat_dict["rs_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, rightside_keywords)
        feat_dict["rs_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, rightside_keywords)
        feat_dict["prcnt_rs_uttered"] = percentage_of_general_keywords_mentioned(
            interview, rightside_keywords)

        # feat_dict["count_ls_rs_switches"] = count_ls_rs_switches(interview)
        return feat_dict

    if photo_split == 'strips':
        # farleft_keywords
        # (These are keywords which only appear on the farleft side of the image)
        farleft_keywords = get_farleft_keyword_set()
        feat_dict["farleft_count"] = count_of_general_keyword(interview, farleft_keywords)
        feat_dict["farleft_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, farleft_keywords)
        feat_dict["farleft_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, farleft_keywords)
        feat_dict["prcnt_farleft_uttered"] = percentage_of_general_keywords_mentioned(
            interview, farleft_keywords)

        # centerleft_keywords
        # (These are keywords which only appear on the centerleft side of the image)
        centerleft_keywords = get_centerleft_keyword_set()
        feat_dict["centerleft_count"] = count_of_general_keyword(interview, centerleft_keywords)
        feat_dict["centerleft_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, centerleft_keywords)
        feat_dict["centerleft_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, centerleft_keywords)
        feat_dict["prcnt_centerleft_uttered"] = percentage_of_general_keywords_mentioned(
            interview, centerleft_keywords)

        # farright_keywords
        # (These are keywords which only appear on the farright side of the image)
        farright_keywords = get_farright_keyword_set()
        feat_dict["farright_count"] = count_of_general_keyword(interview, farright_keywords)
        feat_dict["farright_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, farright_keywords)
        feat_dict["farright_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, farright_keywords)
        feat_dict["prcnt_farright_uttered"] = percentage_of_general_keywords_mentioned(
            interview, farright_keywords)

        # centerright_keywords
        # (These are keywords which only appear on the centerright side of the image)
        centerright_keywords = get_centerright_keyword_set()
        feat_dict["centerright_count"] = count_of_general_keyword(interview, centerright_keywords)
        feat_dict["centerright_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, centerright_keywords)
        feat_dict["centerright_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, centerright_keywords)
        feat_dict["prcnt_centerright_uttered"] = percentage_of_general_keywords_mentioned(
            interview, centerright_keywords)

        return feat_dict

    if photo_split == 'quadrants':
        # NW_keywords
        # (These are keywords which only appear on the centerright side of the image)
        NW_keywords = get_NW_keyword_set()
        feat_dict["NW_count"] = count_of_general_keyword(interview, NW_keywords)
        feat_dict["NW_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, NW_keywords)
        feat_dict["NW_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, NW_keywords)
        feat_dict["prcnt_NW_uttered"] = percentage_of_general_keywords_mentioned(
            interview, NW_keywords)

        # NE_keywords
        # (TheNE are keywords which only appear on the centerright side of the image)
        NE_keywords = get_NE_keyword_set()
        feat_dict["NE_count"] = count_of_general_keyword(interview, NE_keywords)
        feat_dict["NE_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, NE_keywords)
        feat_dict["NE_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, NE_keywords)
        feat_dict["prcnt_NE_uttered"] = percentage_of_general_keywords_mentioned(
            interview, NE_keywords)

        # SE_keywords
        # (TheNE are keywords which only appear on the centerright side of the image)
        SE_keywords = get_SE_keyword_set()
        feat_dict["SE_count"] = count_of_general_keyword(interview, SE_keywords)
        feat_dict["SE_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, SE_keywords)
        feat_dict["SE_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, SE_keywords)
        feat_dict["prcnt_SE_uttered"] = percentage_of_general_keywords_mentioned(
            interview, SE_keywords)

        # SW_keywords
        # (TheNE are keywords which only appear on the centerright side of the image)
        SW_keywords = get_SW_keyword_set()
        feat_dict["SW_count"] = count_of_general_keyword(interview, SW_keywords)
        feat_dict["SW_ty_to_tok_ratio"] = general_keyword_type_to_token_ratio(
            interview, SW_keywords)
        feat_dict["SW_kw_to_w_ratio"]  = general_keyword_to_non_keyword_ratio(
            interview, SW_keywords)
        feat_dict["prcnt_SW_uttered"] = percentage_of_general_keywords_mentioned(
            interview, SW_keywords)

        return feat_dict


def get_cookie_theft_info_unit_features(interview):
    feat_dict = {}
    # Boy IU
    feat_dict["keywordIUSubjectBoy"] = keywordIUSubjectBoy(interview)
    feat_dict["binaryIUSubjectBoy"] = binaryIUSubjectBoy(interview)
    # Girl IU
    feat_dict["keywordIUSubjectGirl"] = keywordIUSubjectGirl(interview)
    feat_dict["binaryIUSubjectGirl"] = binaryIUSubjectGirl(interview)
    # Woman IU
    feat_dict["keywordIUSubjectWoman"] = keywordIUSubjectWoman(interview)
    feat_dict["binaryIUSubjectWoman"] = binaryIUSubjectWoman(interview)
    # Kitchen IU
    feat_dict["keywordIUPlaceKitchen"] = keywordIUPlaceKitchen(interview)
    feat_dict["binaryIUPlaceKitchen"] = binaryIUPlaceKitchen(interview)
    # Exterior IU
    feat_dict["keywordIUPlaceExterior"] = keywordIUPlaceExterior(interview)
    feat_dict["binaryIUPlaceExterior"] = binaryIUPlaceExterior(interview)
    # Cookie IU
    feat_dict["keywordIUObjectCookie"] = keywordIUObjectCookie(interview)
    feat_dict["binaryIUObjectCookie"] = binaryIUObjectCookie(interview)
    # Jar IU
    feat_dict["keywordIUObjectJar"] = keywordIUObjectJar(interview)
    feat_dict["binaryIUObjectJar"] = binaryIUObjectJar(interview)
    # Stool IU
    feat_dict["keywordIUObjectStool"] = keywordIUObjectStool(interview)
    feat_dict["binaryIUObjectStool"] = binaryIUObjectStool(interview)
    # Sink IU
    feat_dict["keywordIUObjectSink"] = keywordIUObjectSink(interview)
    feat_dict["binaryIUObjectSink"] = binaryIUObjectSink(interview)
    # Plate IU
    feat_dict["keywordIUObjectPlate"] = keywordIUObjectPlate(interview)
    feat_dict["binaryIUObjectPlate"] = binaryIUObjectPlate(interview)
    # Dishcloth IU
    feat_dict["keywordIUObjectDishcloth"] = keywordIUObjectDishcloth(interview)
    feat_dict["binaryIUObjectDishcloth"] = binaryIUObjectDishcloth(interview)
    # Water IU
    feat_dict["keywordIUObjectWater"] = keywordIUObjectWater(interview)
    feat_dict["binaryIUObjectWater"] = binaryIUObjectWater(interview)
    # Window IU
    feat_dict["keywordIUObjectWindow"] = keywordIUObjectWindow(interview)
    feat_dict["binaryIUObjectWindow"] = binaryIUObjectWindow(interview)
    # Cupboard IU
    feat_dict["keywordIUObjectCupboard"] = keywordIUObjectCupboard(interview)
    feat_dict["binaryIUObjectCupboard"] = binaryIUObjectCupboard(interview)
    # Dishes IU
    feat_dict["keywordIUObjectDishes"] = keywordIUObjectDishes(interview)
    feat_dict["binaryIUObjectDishes"] = binaryIUObjectDishes(interview)
    # Curtains IU
    feat_dict["keywordIUObjectCurtains"] = keywordIUObjectCurtains(interview)
    feat_dict["binaryIUObjectCurtains"] = binaryIUObjectCurtains(interview)

    # Boy taking IU
    feat_dict["keywordIUActionBoyTaking"] = keywordIUActionBoyTaking(interview)
    feat_dict["binaryIUActionBoyTaking"] = binaryIUActionBoyTaking(interview)

    # Stool falling taking IU
    feat_dict["keywordIUActionStoolFalling"] = keywordIUActionStoolFalling(interview)
    feat_dict["binaryIUActionStoolFalling"] = binaryIUActionStoolFalling(interview)

    # Woman Drying
    feat_dict["keywordIUActionWomanDryingWashing"] = keywordIUActionWomanDryingWashing(interview)
    feat_dict["binaryIUActionWomanDryingWashing"] = binaryIUActionWomanDryingWashing(interview)

    # Water overflowing
    feat_dict["keywordIUActionWaterOverflowing"] = keywordIUActionWaterOverflowing(interview)
    feat_dict["binaryIUActionWaterOverflowing"] = binaryIUActionWaterOverflowing(interview)

    return feat_dict


if __name__ == '__main__':
    pass
