# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np

import time
from datetime import datetime

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
paralleldots.set_api_key("OwoCuujTTYS5o3Vrx0M4A7RevaHLs7uUHlUD3Afa0XI") # laualan@hku.hk
print('Current key:', 'OwoCuujTTYS5o3Vrx0M4A7RevaHLs7uUHlUD3Afa0XI', 1)

keys = [
        'DwGe3alGduDGJnE1FT3O112wMDNJhZDnGysR0KFfZgY', # rlalan@outlook.com
        'JIkATWugt8LIP3PbQrdm6cfm1hZ4DbouVoyw8oAQqhI', # 394414515@qq.com
        'Vf9mBAl7wp0s8Fl43E9aDtVPOZhftIAv0pYJvIlYGD4', # 806124854@qq.com
        '8gAwuDOEE92zakGfDXc6PoCtqhVop0htEnEd4IHFe0U', # yleun.lau@gmail.com
        'xTNXoCxtbg36jCgT6ArKgyOPJtnJdf5QQpEEKYoSwu8', # alanconstantinelau@gmail.com
        'INsEQXjCWuaZJQyzEbHg5Kw6C8liJiHf2DXOHqmHG70', # 122075300@qq.com
        'jpctHohVPtW9QlezbIw4hZ8Ftwh8kXD14U7JlR0WRQo', # 274038499@qq.com
        'qrJoLzvIQqU1ygqtzWDw2JC59zkBsPqSzLLIeRSxLHA', # 562040899@qq.com
        'UtEqIoCXWJN5XiyfgpXJ7gg23LgOu7jVQjl36Xg7UmQ', # wgj0905@hku.hk
	'sHyni9kN5Elhtrmb3Z10f5USPyyn9o0snPCzIqagafM' # liuhuan19951021@163.com
        ]


# %%
df = pd.read_csv('./reindex_df.csv')
print(df.shape)



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
        
def get_stopindex():
    try:
        with open(r'./lyrics_emotion.json', 'r') as fn:
                emo_dict = json.load(fn)
        return max(list(map(lambda e: int(e), list(emo_dict.keys()))))
    except Exception as e:
        return -1
    


# %%
lyric_emo = df['lyrics'].tolist()
index_list = df.index.tolist()

total = len(index_list)
error_count = 0
key_count = 0

for index, lyrics in list(zip(index_list, lyric_emo))[get_stopindex() + 1:]:
    randtime = randint(0, 4)
    try:
        time.sleep(randtime)
        response = paralleldots.emotion(lyrics)
        if 'code' in response and key_count < len(keys):
            key_count += 1
            print('Current key:', keys[key_count], key_count + 1)
            paralleldots.set_api_key(keys[key_count])
            response = paralleldots.emotion(lyrics)
        elif key_count == len(keys):
            print('Stop at', str(index))
            update_error(str(index) + '\tStop.')
            break
        if 'message' in response:
            print('Stop at', str(index))
            update_error(str(index) + '\t' + str(response['message']))
            break
        finished_index = update_emo(index, response)
        print(finished_index, 'done: ', str(round(((index-1)/total)*100, 5)), str(datetime.now()))
    except Exception as e:
        update_error(str(index) + '\t' + str(e))
        error_count += 1
        print(e)
        pass

print(error_count)



