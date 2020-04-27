import json

"""
-rw-r--r--.  1 alan staff   9065637 Apr 27 11:52 lyrics_emotion[0-56818].json
-rw-r--r--.  1 alan staff   1186507 Apr 27 11:42 lyrics_emotion[0-7480].json
-rw-r--r--.  1 alan staff    138228 Apr 27 11:42 lyrics_emotion[0-877].json
-rw-r--r--.  1 alan staff       966 Apr 27 11:42 lyrics_emotion[152444].json
-rw-r--r--.  1 alan staff   2608597 Apr 27 11:42 lyrics_emotion[59147-72341].json
"""

def loadjson(path):
    with open(path, 'r') as f:
        js = json.load(f)
    return js

path1 = r'./lyrics_emotion[0-56818].json'
path2 = r'./lyrics_emotion[59147-72341].json'

js1 = loadjson(path1)
js2 = loadjson(path2)

js1.update(js2)

with open(r'lyric_emotion.json', 'w') as f:
    json.dump(js1, f)
