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
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

from sklearn.utils import shuffle

import json

import paralleldots
paralleldots.set_api_key("OwoCuujTTYS5o3Vrx0M4A7RevaHLs7uUHlUD3Afa0XI")


# %%
df = pd.read_csv('./en_df_sample_30.csv')
df = df.sample(frac=1)
print(df.shape)


# %%
# df.head()

# %% [markdown]
# # Replace Stopwords & Puncutation

# %%
def repl_stp(ly):
    ly_tokenized_words = word_tokenize(ly)
    filtered_stopwords_ly = list(filter(lambda word: word not in stopwords.words('english'), ly_tokenized_words))
    punctutaion_ly = [w for w in filtered_stopwords_ly if not re.fullmatch('[' + string.punctuation + ']+', w)]
    return punctutaion_ly

# %% [markdown]
# # Emotion Detect

# %%
def update_emo(index, emo):
    try:
        with open(r'./lyrics_emotion.json', 'r') as fn:
            emo_dict = json.load(fn)
        emo_dict[index] = emo
    except Exception as e:
        emo_dict = {}
        emo_dict[index] = emo
    with open('./lyrics_emotion.json', 'w') as fp:
        json.dump(emo_dict, fp)
    return index
        
def update_error(index):
    with open(r'./error', 'a') as fn:
        fn.write(str(index) + '\n')


# %%
lyric_emo = df['lyrics'].tolist()
index_list = df.index.tolist()

total = len(index_list)
count = 1
error_count = 0

for index, lyrics in zip(index_list, lyric_emo):
    randtime = randint(0, 4)
    try:
        time.sleep(randtime)
        response = paralleldots.emotion(lyrics)
        finished_index = update_emo(index, response)
        print(finished_index, 'done: ', str(round((count/total)*100, 5)))
        count += 1
    except Exception as e:
        update_error(str(index) + '\t' + str(e))
        error_count += 1
        print(e)

print(error_count)


# # for single sentence
# text="I am trying to imagine you with a personality."
# response=paralleldots.emotion(text)
# print(response)



# # for multiple sentence as array
# text=["I am trying to imagine you with a personality.","This is shit."]
# response=paralleldots.batch_emotion(text)
# print(response)


# %%


