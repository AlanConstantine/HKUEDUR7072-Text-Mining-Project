# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

import time

from random import randint

import re
import string

import nltk


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.utils import shuffle

import json


# %%
df = pd.read_csv('./reindex_df.csv')

df = df[:5000]
print(df.shape)


# %%
# df.head()

# %% [markdown]
# # Replace Stopwords & Puncutation

# %%
def pre_represent(ly):
    ly_tokenized_words = word_tokenize(ly)
    filtered_stopwords_ly = list(filter(lambda word: word not in stopwords.words('english'), ly_tokenized_words)) # removing stopwords
    punctutaion_ly = [stemmer.stem(w) for w in filtered_stopwords_ly if not re.fullmatch('[' + string.punctuation + ']+', w)] # removing puncutation and then stemming
    return ' '.join(punctutaion_ly)


# %%
df['represent'] = df['lyrics'].apply(pre_represent)
df['word_count'] = df['represent'].str.split().str.len()
df = df[df['word_count'] >= 100]
df = df[df['word_count'] <= 1000]


# %%
corpus = df['represent'].tolist()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names())


# %%
df_tfidf['index'] = df_tfidf.index
df = df.merge(df_tfidf, on = 'index', how = 'left').head()
df.to_csv('./represented_lyrics.csv', index = False)


# %%


