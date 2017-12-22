from sqlalchemy import create_engine
import pandas as pd
from itertools import chain
from dementia_classifier.feature_extraction.feature_sets import pos_phrases, pos_syntactic, psycholinguistic, acoustic, discourse
from dementia_classifier.preprocess import get_data
from dementia_classifier import settings
from dementia_classifier.settings import SQL_DBANK_TEXT_FEATURES, SQL_DBANK_DIAGNOSIS, SQL_DBANK_DEMOGRAPHIC, SQL_DBANK_ACOUSTIC_FEATURES, SQL_DBANK_DISCOURSE_FEATURES
# ======================
# setup mysql connection
# ----------------------
from dementia_classifier import db
cnx = db.get_connection()
# ======================


def get_dbank_text_features(datum):
    feat_dict = pos_phrases.get_all(datum)
    feat_dict.update(pos_syntactic.get_all(datum))
    feat_dict.update(psycholinguistic.get_cookie_theft_info_unit_features(datum))
    feat_dict.update(psycholinguistic.get_psycholinguistic_features(datum))
    feat_dict.update(psycholinguistic.get_spatial_features(datum, 'halves'))
    feat_dict.update(psycholinguistic.get_spatial_features(datum, 'strips'))
    feat_dict.update(psycholinguistic.get_spatial_features(datum, 'quadrants'))
    return feat_dict


def save_dementiabank_text_features():
    data = get_data.parse_dementiabank()
    frames = []
    # for key in data:
    for key in data.keys():
        if data[key]:
            print "Processing: %s" % key
            feat_dict = get_dbank_text_features(data[key])
            feat_dict['interview'] = key.replace('.txt', 'c')
            frames.append(feat_dict)
        else:
            print "%s empty, skipping" % key
  
    feat_df = pd.DataFrame(frames)

    # Save to database
    feat_df.to_sql(SQL_DBANK_TEXT_FEATURES, cnx, if_exists='replace', index=False)


def save_diagnosis():
    diagnosis = pd.read_csv(settings.DBANK_DIAGNOSIS, sep=' ')
    diagnosis.to_sql(SQL_DBANK_DIAGNOSIS, cnx, if_exists='replace', index=False)


def save_demographic():
    demographic = pd.read_csv(settings.DBANK_AGE_GENDER, sep=' ')
    
    # Impute missing values with average male and female age
    male_avg    = demographic[demographic['gender'] == 'male'].age.mean()
    female_avg  = demographic[demographic['gender'] == 'female'].age.mean()
    
    male        = demographic[demographic['gender'] == 'male'].fillna(male_avg)
    female      = demographic[demographic['gender'] == 'female'].fillna(female_avg)
    
    demographic = pd.concat([male, female])
    demographic.to_sql(SQL_DBANK_DEMOGRAPHIC, cnx, if_exists='replace', index=False)

    
def save_acoustic():
    # Extract control acoustic features
    control  = acoustic.get_all(settings.SOUNDFILE_CONTROL_DATA_PATH)
    df_control = pd.DataFrame.from_dict(control, orient="index")

    # Extract dementia acoustic features
    dementia  = acoustic.get_all(settings.SOUNDFILE_DEMENTIA_DATA_PATH)
    df_dementia = pd.DataFrame.from_dict(dementia, orient="index")

    # Merge dfs
    feat_df = pd.concat([df_dementia, df_control])

    # Save interview field for joins
    feat_df["interview"] = feat_df.index
    feat_df.reset_index(inplace=True)

    # Save to sql
    feat_df.to_sql(SQL_DBANK_ACOUSTIC_FEATURES, cnx, if_exists='replace', index=False)


def save_discourse():
    # Extract control discourse features
    control  = discourse.get_all(settings.DISCOURSE_CONTROL_DATA_PATH)
    df_control = pd.DataFrame.from_dict(control, orient="index")

    # Extract dementia discourse features
    dementia  = discourse.get_all(settings.DISCOURSE_DEMENTIA_DATA_PATH)
    df_dementia = pd.DataFrame.from_dict(dementia, orient="index")

    # Merge dfs
    feat_df = pd.concat([df_dementia, df_control])

    # Save interview field for joins
    feat_df["interview"] = feat_df.index
    feat_df['interview'] = feat_df['interview'] + 'c'  # So it's consistent with sound files

    # Save to sql
    feat_df.to_sql(SQL_DBANK_DISCOURSE_FEATURES, cnx, if_exists='replace', index=False)


def save_all_to_sql():
    # Save all dementiabank_text_features
    save_dementiabank_text_features()
    save_acoustic()
    save_discourse()
    save_diagnosis()
    save_demographic()

