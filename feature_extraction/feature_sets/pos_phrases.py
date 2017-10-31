# takes in a list of string and turns them into a list of features
from __future__ import division
import nltk
from collections import defaultdict
from collections import Counter
from pos_syntactic import build_tree
import math


"""
=============================================================

HELPER FUNCTIONS

=============================================================
"""


# input: NLP object for one paragraph
# returns: Returns length of phrases in utterance w.r.t. number of words
def getPhraseLength(nlp_obj, phrase_type):

    def count(node, multiplier):

        if node.key == phrase_type:
            multiplier += 1

        # its a word!
        if node.phrase:

            return multiplier * len(node.phrase.split(' '))

        phrase_length = 0

        for child in node.children:
            phrase_length += count(child, multiplier)

        return phrase_length

    # build the syntactic tree

    Phrase_length = 0

    for tree in nlp_obj['parse_tree']:

        root = build_tree(tree)
        multiplier = 0

        if root.key == phrase_type:
            multiplier += 1

        for child in root.children:
            Phrase_length = count(child, multiplier)

    return Phrase_length


# input: NLP object for one paragraph
# returns: Returns count of phrases in utterance with embedded phrases of the
# same type included in the calculation
def getPhraseCountEmbedded(nlp_obj, phrase_type):

    def count(node):

        phrase_count = 0

        if node.key == phrase_type:
            phrase_count += 1

        # its a word!
        if node.phrase:
            return phrase_count

        for child in node.children:
            phrase_count += count(child)

        return phrase_count

    Phrase_count = 0
    for tree in nlp_obj['parse_tree']:
        # build the syntactic tree
        root = build_tree(tree)

        if root.key == phrase_type:
            Phrase_count += 1

        for child in root.children:
            Phrase_count += count(child)

    return Phrase_count


# input: NLP object for one paragraph
# returns: Returns count of phrases in utterance so only the largest phrase parent counts
# but not its children
def getPhraseCountNonEmbedded(nlp_obj, phrase_type):

    def count(node):

        # We've hit our phrase type and can backtrac
        if node.key == phrase_type:
            return 1

        else:

            # its a word!
            if node.phrase:
                return 0

            phrase_count = 0

            for child in node.children:
                phrase_count += count(child)

            return phrase_count

    Phrase_count = 0
    # build the syntactic tree
    for tree in nlp_obj['parse_tree']:

        root = build_tree(tree)

        if root.key == phrase_type:
            Phrase_count += 1

        for child in root.children:
            Phrase_count += count(child)

    return Phrase_count


"""
=============================================================

WORD TYPE COUNTS

=============================================================
"""


# input: NLP object for one paragraph
# returns: number of normalized nouns in text
def getNumNouns(nlp_obj):

    pos_freq = nlp_obj['pos_freq']

    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS'] + pos_freq['NNPS']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized verbs in text
def getNumVerbs(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized inflected verbs in text
def getNumInflectedVerbs(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ']) / pos_freq['SUM']

# input: NLP object for one paragraph
# returns: number of normalized determiners in text


def getNumDeterminers(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['DT'] + pos_freq['PDT'] + pos_freq['WDT']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized adverbs in text
def getNumAdverbs(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['RB'] + pos_freq['RBR'] + pos_freq['RBS'] + pos_freq['WRB']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized adjectives in text
def getNumAdjectives(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['JJ'] + pos_freq['JJR'] + pos_freq['JJS']) / pos_freq['SUM']

# input: NLP object for one paragraph
# returns: number of normalized interjections in text


def getNumInterjections(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    pos_freq = nlp_obj['pos_freq']
    return (pos_freq['UH']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized subordinate conjunctions in text
def getNumSubordinateConjunctions(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['IN']) / pos_freq['SUM']


# input: NLP object for one paragraph
# returns: number of normalized coordinate conjunctions in text
def getNumCoordinateConjunctions(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    if pos_freq['SUM'] == 0:
        return 0
    return (pos_freq['CC']) / pos_freq['SUM']


"""
===========================================================

WORD TYPE RATIOS

===========================================================
"""


# input: NLP object for one paragraph
# returns: ratio of nouns to verbs in the paragraph
def getRatioVerb(nlp_obj):

    pos_freq = nlp_obj['pos_freq']

    return (pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS'] + pos_freq['NNPS']) / (pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ'] + 1)


# input: NLP object for one paragraph
# returns: ratio of nouns to verbs in the paragraph
def getRatioNoun(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    num_nouns = pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS'] + pos_freq['NNPS']
    num_verbs = pos_freq['VB'] + pos_freq['VBD'] + pos_freq['VBG'] + pos_freq['VBN'] + pos_freq['VBP'] + pos_freq['VBZ']

    return num_nouns / (num_nouns + num_verbs + 1)


# input: NLP object for one paragraph
# returns: ratio of pronouns to nouns in the paragraph
def getRatioPronoun(nlp_obj):

    pos_freq = nlp_obj['pos_freq']
    num_nouns = pos_freq['NN'] + pos_freq['NNP'] + pos_freq['NNS'] + pos_freq['NNPS']

    num_pronouns = pos_freq['PRP'] + pos_freq['PRP$'] + pos_freq['PRP'] + pos_freq['WHP'] + pos_freq['WP$']

    return num_pronouns / (num_nouns + 1)


# input: NLP object for one paragraph
# returns: ratio of coordinate- to subordinate conjunctions in the paragraph
def getRatioCoordinate(nlp_obj):

    pos_freq = nlp_obj['pos_freq']

    return pos_freq['CC'] / (pos_freq['IN'] + 1)


# input: NLP object for one paragraph
# returns: ratio of  types to tokens
def getTTR(nlp_obj):

    num_types = len(set(nlp_obj['token']))
    num_words = len(nlp_obj['token'])

    return num_types / num_words


# input: NLP object for one paragraph
# returns: average ratio of types to tokens using a sliding window
def getMATTR(nlp_obj):

    window = 20
    total_len = len(nlp_obj['token'])

    words_table = Counter(nlp_obj['token'][0:window])
    uniq = len(set(words_table))

    moving_ttr = list([uniq / window])

    for i in range(window, total_len):

        word_to_remove = nlp_obj['token'][i - window]
        words_table[word_to_remove] -= 1

        if words_table[word_to_remove] == 0:

            uniq -= 1

        next_word = nlp_obj['token'][i]
        words_table[next_word] += 1

        if words_table[next_word] == 1:

            uniq += 1

        moving_ttr.append(uniq / window)

    return sum(moving_ttr) / len(moving_ttr)


# input: NLP object for one paragrah
# returns: Brunet index for that paragraph
def getBrunetIndex(nlp_obj):

    # number of word types
    word_types = len(set(nlp_obj['token']))

    # number of words
    words = len(nlp_obj['token'])

    # Brunet's index
    return words**(word_types * -0.165)

# input: NLP object for one paragrah
# returns: Honore statistic for that paragraph


def getHonoreStatistic(nlp_obj):

    # number of word types
    word_types = len(set(nlp_obj['token']))

    # number of words
    words = len(nlp_obj['token'])

    words_table = Counter(nlp_obj['token'])

    words_occuring_once = len([word for word in nlp_obj['token'] if words_table[word] == 1])

    # unlikely case
    if word_types == 0:
        return 0

    if words_occuring_once / word_types == 1:
        return (100 * math.log(words)) / (2 - words_occuring_once / word_types)

    return (100 * math.log(words)) / (1 - words_occuring_once / (word_types))


# input: NLP object for one paragrah
# returns: Mean word length
def getMeanWordLength(nlp_obj):

    tokens = nlp_obj['token']

    word_length = [len(word) for word in tokens]

    # Just to prevent crash
    if len(tokens) == 0:
        return 0

    return sum(word_length) / len(tokens)

# input: NLP object for one paragrah
# returns: number of NID words (length > 2) in paragraph


def getNumberOfNID(nlp_obj):

    pos_tag = nlp_obj['pos']

    foreign_words = [word_pos for word_pos in pos_tag if len(word_pos[0]) > 2 and word_pos[1] == 'FW']

    return len(foreign_words)

# input: NLP object for one paragraph
# returns: normalized number of "uh" and "um"


def getDisfluencyFrequency(nlp_obj):

    tokens = nlp_obj['token']

    um_uh_words = [word for word in tokens if word == 'um' or word == 'uh']

    # just to prevent crash
    if len(tokens) == 0:
        return 0

    return len(um_uh_words) / len(tokens)


# input: NLP object for one paragraph
# returns: Get total number of words excluding NID and filled pauses
def getTotalNumberOfWords(nlp_obj):

    tokens = nlp_obj['token']
    pos_tag = nlp_obj['pos']

    foreign_words = [word_pos for word_pos in pos_tag if word_pos[1] == 'FW']
    um_uh_words = [word for word in tokens if word == 'um' or word == 'uh']

    return len(tokens) - len(foreign_words) - len(um_uh_words)

# input: NLP object for one paragraph
# returns: Returns mean length of sentence w.r.t. number of words


def getMeanLengthOfSentence(nlp_obj):

    raw_text = nlp_obj['raw']
    tokens = nlp_obj['token']
    n_sentences = len(nltk.tokenize.sent_tokenize(raw_text))
    n_words = len(tokens)

    # just to prevent crash
    if n_words == 0:
        return 0

    return n_sentences / n_words

"""

======================================================

 PHRASE TYPE FEATURES

======================================================

"""

# input: NLP object for one paragraph
# returns: Returns proportion of noun phrases in utterance w.r.t. number of words


def getNPProportion(nlp_obj):

    word_count = len(nlp_obj['token'])

    # Prevent crash
    if word_count == 0:
        return 0

    return getPhraseLength(nlp_obj, 'NP') / word_count


# input: NLP object for one paragraph
# returns: Returns proportion of verb phrases in utterance w.r.t. number of words
def getVPProportion(nlp_obj):

    word_count = len(nlp_obj['token'])

    # Prevent crash
    if word_count == 0:
        return 0
    return getPhraseLength(nlp_obj, 'VP') / word_count

# input: NLP object for one paragraph
# returns: Returns proportion of preposition phrases in utterance w.r.t. number of words


def getPProportion(nlp_obj):

    word_count = len(nlp_obj['token'])

    # Prevent crash
    if word_count == 0:
        return 0
    return getPhraseLength(nlp_obj, 'PP') / word_count

# input: NLP object for one paragraph
# returns: Returns average length (in words) of noun phrases in utterance w.r.t. number of noun phrases
# This is embedded so subphrases are also counted


def getAvgNPTypeLengthEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'NP')

    phrase_count = getPhraseCountEmbedded(nlp_obj, 'NP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count

# input: NLP object for one paragraph
# returns: Returns average length (in words) of verb phrases in utterance w.r.t. number of verb phrases
# This is embedded so subphrases are also counted


def getAvgVPTypeLengthEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'VP')

    phrase_count = getPhraseCountEmbedded(nlp_obj, 'VP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count

# input: NLP object for one paragraph
# returns: Returns average length (in words) of prepositional phrases in utterance w.r.t. number of prepositional phrases
# This is embedded so subphrases are also counted


def getAvgPPTypeLengthEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'PP')

    phrase_count = getPhraseCountEmbedded(nlp_obj, 'PP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count


# input: NLP object for one paragraph
# returns: Returns average length (in words) of noun phrases in utterance w.r.t. number of noun phrases
# This is non-embedded so only the largest phrase type is counted
def getAvgNPTypeLengthNonEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'NP')

    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'NP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count


# input: NLP object for one paragraph
# returns: Returns average length (in words) of verb phrases in utterance w.r.t. number of verb phrases
# This is non-embedded so only the largest phrase type is counted
def getAvgVPTypeLengthNonEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'VP')

    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'VP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count


# input: NLP object for one paragraph
# returns: Returns average length (in words) of prepositional phrases in utterance w.r.t. number of prepositional phrases
# This is non-embedded so only the largest phrase type is counted
def getAvgPPTypeLengthNonEmbedded(nlp_obj):

    # phrase length in words summed up
    phrase_length = getPhraseLength(nlp_obj, 'PP')

    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'VP')

    # Prevent crash
    if phrase_count == 0:
        return 0

    return phrase_length / phrase_count

# input: NLP object for one paragraph
# returns: Returns number of noun phrases divided by the number of words in the sentence
# ATTENTION we use the nonembbeded count here


def getNPTypeRate(nlp_obj):

    word_count = len(nlp_obj['token'])
    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'NP')

    # Prevent crash
    if word_count == 0:
        return 0

    return phrase_count / word_count


# input: NLP object for one paragraph
# returns: Returns number of verb phrases divided by the number of words in the sentence
# ATTENTION we use the nonembbeded count here
def getVPTypeRate(nlp_obj):

    word_count = len(nlp_obj['token'])
    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'VP')

    # Prevent crash
    if word_count == 0:
        return 0

    return phrase_count / word_count


# input: NLP object for one paragraph
# returns: Returns number of prepositional phrases  divided by the number of words in the sentence
# ATTENTION we use the nonembbeded count here
def getPPTypeRate(nlp_obj):

    word_count = len(nlp_obj['token'])
    phrase_count = getPhraseCountNonEmbedded(nlp_obj, 'PP')

    # Prevent crash
    if word_count == 0:
        return 0

    return phrase_count / word_count


# input: list of utterances for one interview stored as [{},{},{}]
# returns: list of features for  interview
def get_all(interview):
    features = {}
    # POS counts
    features["NumNouns"] = sum([getNumNouns(utterance) for utterance in interview]) / len(interview)
    features["NumVerbs"] = sum([getNumVerbs(utterance) for utterance in interview])
    features["MATTR"] = sum([getMATTR(utterance) for utterance in interview]) / len(interview)
    features["BrunetIndex"] = sum([getBrunetIndex(utterance) for utterance in interview]) / len(interview)
    features["HonoreStatistic"] = sum([getHonoreStatistic(utterance) for utterance in interview]) / len(interview)

    # Summary statistics
    features["NumberOfNID"] = sum([getNumberOfNID(utterance) for utterance in interview]) / len(interview)
    features["MeanWordLength"] = sum([getMeanWordLength(utterance) for utterance in interview]) / len(interview)
    features["TotalNumberOfWords"] = sum([getTotalNumberOfWords(utterance) for utterance in interview]) / len(interview)
    features["DisfluencyFrequency"] = sum([getDisfluencyFrequency(utterance)
                                           for utterance in interview]) / len(interview)
    features["MeanLengthOfSentence"] = sum([getMeanLengthOfSentence(utterance)
                                            for utterance in interview]) / len(interview)

    # Phrase features/len(interview)
    features["NumAdverbs"] = sum([getNumAdverbs(utterance) for utterance in interview]) / len(interview)
    features["NumAdjectives"] = sum([getNumAdjectives(utterance) for utterance in interview]) / len(interview)
    features["NumDeterminers"] = sum([getNumDeterminers(utterance) for utterance in interview]) / len(interview)
    features["NumInterjections"] = sum([getNumInterjections(utterance) for utterance in interview]) / len(interview)
    features["NumInflectedVerbs"] = sum([getNumInflectedVerbs(utterance) for utterance in interview]) / len(interview)
    features["NumCoordinateConjunctions"] = sum([getNumCoordinateConjunctions(utterance)
                                                 for utterance in interview]) / len(interview)
    features["NumSubordinateConjunctions"] = sum(
        [getNumSubordinateConjunctions(utterance) for utterance in interview]) / len(interview)

    # POS ratios
    features["RatioNoun"] = sum([getRatioNoun(utterance) for utterance in interview]) / len(interview)
    features["RatioVerb"] = sum([getRatioVerb(utterance) for utterance in interview]) / len(interview)
    features["RatioPronoun"] = sum([getRatioPronoun(utterance) for utterance in interview]) / len(interview)
    features["RatioCoordinate"] = sum([getRatioCoordinate(utterance) for utterance in interview]) / len(interview)

    # Weird statistics
    features["TTR"] = sum([getTTR(utterance) for utterance in interview]) / len(interview)
    features["NPTypeRate"] = sum([getNPTypeRate(utterance) for utterance in interview]) / len(interview)
    features["VPTypeRate"] = sum([getVPTypeRate(utterance) for utterance in interview]) / len(interview)
    features["PPTypeRate"] = sum([getPPTypeRate(utterance) for utterance in interview]) / len(interview)
    features["PProportion"] = sum([getPProportion(utterance) for utterance in interview]) / len(interview)
    features["NPProportion"] = sum([getNPProportion(utterance) for utterance in interview]) / len(interview)
    features["VPProportion"] = sum([getVPProportion(utterance) for utterance in interview]) / len(interview)
    features["AvgNPTypeLengthEmbedded"] = sum([getAvgNPTypeLengthEmbedded(utterance)
                                               for utterance in interview]) / len(interview)
    features["AvgVPTypeLengthEmbedded"] = sum([getAvgVPTypeLengthEmbedded(utterance)
                                               for utterance in interview]) / len(interview)
    features["AvgPPTypeLengthEmbedded"] = sum([getAvgPPTypeLengthEmbedded(utterance)
                                               for utterance in interview]) / len(interview)
    features["AvgPPTypeLengthNonEmbedded"] = sum(
        [getAvgPPTypeLengthNonEmbedded(utterance) for utterance in interview]) / len(interview)
    features["AvgNPTypeLengthNonEmbedded"] = sum(
        [getAvgNPTypeLengthNonEmbedded(utterance) for utterance in interview]) / len(interview)
    features["AvgVPTypeLengthNonEmbedded"] = sum(
        [getAvgVPTypeLengthNonEmbedded(utterance) for utterance in interview]) / len(interview)

    return features

# For testing
#------------------------------------------------

# if __name__ == '__main__':
# 	s0 = "this little boy here is taking cookies "
# 	s1 = " This is a second sentence "
# 	s2 = "This. Sentence has punctuation!"
# 	s3 = "And this sentsce has spelling mistkaes"
# 	s4 = "this little boy here is also taking cookies "
# 	s5  = "An elephant fish pork monkey"
# 	l = [s0, s1, s2, s3, s4, s5]
# 	print 'avg_cos_dist', avg_cos_dist(l)
# 	print 'proportion_below_threshold', proportion_below_threshold(l,0)
