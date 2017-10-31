from sqlalchemy import create_engine
import pandas as pd
from itertools import chain
from dementia_classifier.feature_extraction.feature_sets import pos_phrases, pos_syntactic, psycholinguistic, acoustic, discourse
from dementia_classifier.preprocess import get_data

# ======================
# setup mysql connection
# ----------------------
from dementia_classifier import db
cnx = db.get_connection()
# ======================


def get_blog_name(url):
    return url.replace('http://', '').replace('https://', '').replace('.blogspot.ca', '')


def get_blog_text_features(datum):
    feat_dict = pos_phrases.get_all(datum)
    feat_dict.update(pos_syntactic.get_all(datum))
    feat_dict.update(psycholinguistic.get_psycholinguistic_features(datum))
    return feat_dict


def process_blog(blog, name):
    posts = []
    for post_id in blog:
        post = blog[post_id]
        if post:
            print 'Processing %s' % post_id
            feat_dict = pos_phrases.get_all(post)
            feat_dict.update(pos_syntactic.get_all(post))
            feat_dict.update(psycholinguistic.get_psycholinguistic_features(post))
            feat_dict["number_of_sentences"] = len(post)
            feat_dict['blog'] = name
            feat_dict['id'] = post_id
            posts += [feat_dict]
    return posts
    

def in_database(name):
    try:
        if pd.read_sql_table(name, cnx, chunksize=1):
            return True
    except ValueError:
        return False


def save_all_blogs():
    data = get_data.parse_blogs()
    for blog in data:
        print 'Processing %s' % blog
        name = get_blog_name(blog)
        sqlname = "%s_text_features" % name
        if not in_database(sqlname):
            posts = process_blog(data[blog], name)
            df = pd.DataFrame(posts)
            df.to_sql(sqlname, cnx, if_exists='replace', index=False)


def save_blog_quality():
    qual = get_data.get_blog_quality()
    df = pd.DataFrame(qual)
    # Fix blog name
    df['blog'] = df['blog'].apply(get_blog_name)
    df.to_sql("blog_quality", cnx, if_exists='replace', index=False)


def save_blog_data():
    save_all_blogs()
    save_blog_quality()



if __name__ == '__main__':
    import pdb; pdb.set_trace()
