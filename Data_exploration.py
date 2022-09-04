# data processing libraries
import pandas as pd

# display wider columns in pandas data frames where necessary
pd.set_option('max_colwidth',150)

import spacy
nlp = spacy.load("en_core_web_sm")

# supporting libraries
import re
import pickle
def dataExplore():
    # file location of the data
    input_folder = '../data/MSRP/MSRParaphraseCorpus/'
    output_folder = '../output/lda/'

    file_name = 'msr-para-val.tsv'
    # load data
    df_data = pd.read_csv("../data/MSRP/MSRParaphraseCorpus/msr-para-val.tsv", sep="\t", quoting=3)

    # display first row of the data frame
    # print(df_data.shape)
    # print(df_data.head(1).T)

    # select ONLY data with specified section and publication and non-duplicated texts of article
    df_data['#1 String'] = df_data['#1 String'].fillna("")
    df_data = df_data[df_data['#1 String'].apply(len) > 0]

    df_data['#2 String'] = df_data['#2 String'].fillna("")
    df_data = df_data[df_data['#2 String'].apply(len) > 0]

    df_data = df_data.drop_duplicates('#1 String')

    df_data.shape

    # Publications in the data
    print('Number of unique values:')
    df = df_data.groupby('#1 String')[['#1 ID', '#2 ID']].nunique()

    selected_publications = list(df.index)
    # print(len(selected_publications))

    df_data = df_data[df_data['#1 String'].isin(selected_publications)]
    # print(df_data.shape)

    # clean text
    df_data['#1 String'] = df_data['#1 String'].str.replace(r"[^A-Za-z0-9//-/.,!?:; ]", '', regex=True)

    # select texts that have at least 500 but no more than 10000 symbols
    df_data['text_length_1'] = df_data['#1 String'].fillna("").apply(len)
    df_data = df_data[df_data['text_length_1'] >= 50]
    df_data = df_data[df_data['text_length_1'] < 10000]

    # cut text to have no more than 1500 symbols
    df_data['#1 String'] = df_data['#1 String'].str[:1500]
    # print(df_data.shape)

    # checking number of articles per section
    s = df_data['#1 String'].value_counts()
    s.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    print(s)

    for k in range(3):
        file_name = 'data_part_' + str(k) + '.pickle'

        # load data
        with open(output_folder + file_name, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            df_data = pickle.load(f)

        # get spaCy doc
        print(k)
        # % time
        df_data['spacy_doc'] = df_data['#1 String'].apply(lambda x: nlp(x))
        print("=" * 50)

        # delete text of article
        del df_data['#1 String']
        data= df_data['#1 String']
    return data
