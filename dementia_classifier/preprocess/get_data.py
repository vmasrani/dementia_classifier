# encoding: utf-8
import os
import sys
import re
import requests
import nltk
from collections import defaultdict
import util
import nltk.data
import xml.etree.ElementTree as ET
from make_blog_corpus import create_corpus as create_blog_corpus
from dementia_classifier import settings
try:
    import cPickle as pickle
except:
    import pickle

# takes a long string and cleans it up and converts it into a vector to be extracted
# NOTE: Significant preprocessing was done by sed - make sure to run this script on preprocessed text

# Data structure
# data = {
# 		"id1":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]}, <--single utterance
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 	],													  <--List of all utterances made during interview
# 		"id2":[	{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 		{'pos':[], 'raw':[], 'tokens':[], 'pos_freq':[], 'parse_tree':[]},
# 		 	],
# 		...
# }


def get_stanford_parse(sentence, port=9000):
    # raw = sentence['raw']
    # We want to iterate through k lines of the file and segment those lines as a session
    # pattern = '[a-zA-Z]*=\\s'
    # re.sub(pattern, '', raw)
    re.sub(r'[^\x00-\x7f]', r'', sentence)
    sentence = util.remove_control_chars(sentence)
    try:
        r = requests.post('http://localhost:' + str(port) +
                          '/?properties={\"annotators\":\"parse\",\"outputFormat\":\"json\"}', data=sentence)
    except requests.exceptions.ConnectionError, e:
        print "We received the following error in get_data.get_stanford_parse():"
        print e
        print "------------------"
        print 'Did you start the Stanford server? If not, try:\n java -Xmx4g -cp "lib/stanford/stanford-corenlp-full-2015-12-09/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 20000'
        print "------------------"
        sys.exit(1)

    json_obj = r.json()
    return json_obj['sentences'][0]


def _processUtterance(uttr):
    uttr = util.clean_uttr(uttr)         # clean
    tokens = nltk.word_tokenize(uttr)    # Tokenize
    tagged_words = nltk.pos_tag(tokens)  # Tag

    # Get the frequency of every type
    pos_freq = defaultdict(int)
    for word, wordtype in tagged_words:
        pos_freq[wordtype] += 1

    pos_freq['SUM'] = len(tokens)
    pt_list = []
    bd_list = []
    for u in util.split_string_by_words(uttr, settings.PARSER_MAX_LENGTH):
        if u is not "":
            stan_parse = get_stanford_parse(u)
            pt_list.append(stan_parse["parse"])
            bd_list.append(stan_parse["basic-dependencies"])
    datum = {"pos": tagged_words, "raw": uttr, "token": tokens,
             "pos_freq": pos_freq, "parse_tree": pt_list, "basic_dependencies": bd_list}
    return datum


# Extract data from dbank directory
def _parse_dementiabank(filepath):
    parsed_data = {}
    for filename in os.listdir(filepath):
        if filename.endswith(".txt"):
            with open(os.path.join(filepath, filename)) as file:
                print "Parsing: " + filename
                session_utterances = []
                for line in file:
                    uttr = util.clean_uttr(line)
                    if util.isValid(uttr):
                        session_utterances.append(_processUtterance(uttr))
                parsed_data[filename] = session_utterances  # Add session
    else:
        print "Filepath not found: " + filepath
        print "Data may be empty"
    return parsed_data


# All samples are processed and  cached to pickle after creating
# Diagnoses come from dementia_classifier/data/diag.txt
def parse_dementiabank():
    if not os.path.isfile(settings.DBANK_PICKLE_PATH):
        data = _parse_dementiabank(settings.DBANK_DATA_PATH)
        with open(settings.DBANK_PICKLE_PATH, 'wb') as f:
            pickle.dump(data, f)
    else:
        print 'Loading dbank pickle from: %s' % settings.DBANK_PICKLE_PATH
        with open(settings.DBANK_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)

    return data


def _parse_blog(blog):
    parsed_data = {}
    for post in blog:
        post_id = post.attrib['id']
        processed_post = []
        for sentence in post:
            sentence = sentence.text
            if util.isValid(sentence):
                try:
                    processed_post.append(_processUtterance(sentence))
                except ValueError:
                    print "Cannot parse: %s" % sentence
        parsed_data[post_id] = processed_post
    return parsed_data


def parse_blogs():
    if not os.path.isfile(settings.BLOG_PICKLE_PATH):
        # If corpus is not downloaded and saved to XML, do that
        if not os.path.isfile(settings.BLOG_CORPUS_PATH):
            create_blog_corpus(settings.BLOG_CORPUS_PATH)

        # Traverse XML and process each blog
        tree = ET.parse(settings.BLOG_CORPUS_PATH)
        root = tree.getroot()
        blogs = {}
        for blog in root:
            name = blog.attrib['name']
            print "Processing %s"  % name
            blogs[name] = _parse_blog(blog)

        # Save to pickle
        with open(settings.BLOG_PICKLE_PATH, 'wb') as f:
            pickle.dump(blogs, f)
    else:
        print 'Loading blog pickle from: %s' % settings.BLOG_PICKLE_PATH
        with open(settings.BLOG_PICKLE_PATH, "rb") as f:
            blogs = pickle.load(f)

    return blogs


def get_blog_quality():
    # If corpus is not downloaded and saved to XML, do that
    if not os.path.isfile(settings.BLOG_CORPUS_PATH):
        create_blog_corpus(settings.BLOG_CORPUS_PATH)

    # Traverse XML and extract quality
    tree = ET.parse(settings.BLOG_CORPUS_PATH)
    root = tree.getroot()
    quality = []
    
    for blog in root:
        name = blog.attrib['name']
        print "Processing %s"  % name
        for post in blog:
            post_id = post.attrib['id']
            date = post.attrib['date']
            qual = post.attrib['quality']
            quality += [{'date': date, 'quality': qual, 'id': post_id, 'blog': name}]
    
    return quality


def debug():
    data = _parse_dementiabank('dementia_classifier/data/test/dementia')
    return data


if __name__ == '__main__':
    parse_blogs()
