# Lau Alan
# Extraction lyrics Emotions
# 2020.03.14

import numpy as np
import pandas as pd

from langdetect import detect

import seaborn as sns
# load data

df = pd.read_csv('./lyrics.csv')
print(df.shape)

print(df.columns)

df = df.replace({'\n': ' '}, regex=True)

df['word_count'] = df['lyrics'].str.split().str.len()
df_clean = df[df['word_count'] >= 100]
df_clean = df[df['word_count'] <= 1000]


def detect_en(lyrics):
    try:
        if detect(lyrics) == 'en':
            return True
        else:
            return False
    except Exception as e:
        return False

df_clean['if_en'] = df_clean['lyrics'].apply(detect_en)


sns_violinplot_word_count = sns.violinplot(x=df_clean["word_count"])
sns_violinplot_word_count.savefig("./word_count_violinplot.png")
fig = sns_violinplot_word_count.get_figure()
fig.savefig("./word_count_violinplot.png")

df_clean.to_csv('./filtered_en_lyrics.csv', index=False)
