# encoding: utf-8
from nltk.corpus import wordnet as wn
import nltk
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer


DISFLUENCIES = ["uh", "um", "er", "ah"]

control_chars = ''.join(map(unichr, range(0, 32) + range(127, 160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
printable = set(string.printable)
lmtzr = WordNetLemmatizer()


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    # Return noun by default
    return wn.NOUN


def isValid(inputString):
    # Line should not contain numbers
    if(any(char.isdigit() for char in inputString)):
        return False
    # Line should not be empty
    elif not inputString.strip():
        return False
    # Line should contain characters (not only consist of punctuation)
    elif not bool(re.search('[a-zA-Z]', inputString)):
        return False
    else:
        return True


def sentences(data):
    # Filter non-ascii
    data = filter(lambda x: x in printable, data)
    sentences = sent_detector.tokenize(data.strip())
    return sentences


def remove_disfluencies(uttr):
    tokens = nltk.word_tokenize(uttr)
    tmp = [t for t in tokens if t.lower() not in DISFLUENCIES]
    clean = " ".join(tmp)
    if isValid(clean):
        return clean
    else:
        return ""

# Lemmatize


def lemmetize(uttr):

    tokens = nltk.word_tokenize(uttr)
    tagged_words = nltk.pos_tag(tokens)

    lemmed_words = []
    for word, wordtype in tagged_words:
        wt = penn_to_wn(wordtype)
        lem = lmtzr.lemmatize(word, wt)
        lemmed_words.append(lem)

    return " ".join(lemmed_words)


# Clean uttr / Remove non ascii
def clean_uttr(uttr):
    uttr = uttr.decode('utf-8', 'ignore').strip()
    uttr = uttr.encode('utf-8')
    uttr = re.sub(r'[^\x00-\x7f]', r'', uttr)
    return uttr


def split_string_by_words(sen, n):
    tokens = sen.split()
    return [" ".join(tokens[(i) * n:(i + 1) * n]) for i in range(len(tokens) / n + 1)]


def remove_control_chars(s):
    return control_char_re.sub('', s)
