# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


from time import time

from tqdm import tqdm

import pickle
import json

# %% [markdown]
# # Clustering

# %%
K = list(range(2, 10))


# %%
df = pd.read_csv(r'./df_tfidf4395.csv')
# df.head()


# %%
features = df.columns.tolist()[:-1]


# %%
data = df[features].values

# %% [markdown]
# ## Kmeans

# %%
results = {}
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=905).fit(data)
    labels = kmeans.labels_
    results[k] = {
                  'db_index': davies_bouldin_score(data, labels),
                  'labels': kmeans.labels_,
                  'centres': kmeans.cluster_centers_
                 }
    print(k, davies_bouldin_score(data, labels))

# %% [markdown]
# ## LDA

# %%
n_top_words = 20


# %%
lda = LatentDirichletAllocation(n_components=4, 
                                max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=905)


# %%
t0 = time()
lda.fit(data)
print("done in %0.3fs." % (time() - t0))


# %%
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# %%
print_top_words(lda, features, n_top_words)

# %% [markdown]
# # Classification
# %% [markdown]
# ## Data preparing

# %%
with open(r'./lyrics_emotion[0-7480].json', 'r') as fn:
    emo_dict = json.load(fn)


# %%
data_cls = pd.read_csv('./df_4395.csv')
# data_cls.head()


# %%
data_cls['clustered_label'] = pd.Series(results[4]['labels'])
# data_cls.head()


# %%
emotions = ['Fear',
            'Sad',
            'Bored',
            'Happy',
            'Excited',
            'Angry']


# %%
for e in tqdm(emotions):
    data_cls[e] = np.nan


# %%
for i in tqdm(range(len(data_cls))):
    lyrics_emotion = emo_dict[str(i)]['emotion']
    values = [lyrics_emotion[e] for e in emotions]
    data_cls.at[i, emotions] = values
#     break

# %% [markdown]
# ## Modeling
# %% [markdown]
# ### Measures

# %%
def measures(ypred, ytest):
    return f1_score(ypred, ytest, average='micro'), cohen_kappa_score(ypred, ytest)

kappa_scorer = make_scorer(cohen_kappa_score)

scoring = kappa_scorer

# %% [markdown]
# ### SVM

# %%
def svm_model(xtrain, xtest, ytrain, ytest, batch):
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    clf = GridSearchCV(
        SVC(), 
        param_grid, 
        scoring=scoring, 
        n_jobs=-1, 
        cv=10
    )
    searcher = clf.fit(xtrain, ytrain)
    estimator = searcher.best_estimator_
    f1, kappa = measures(estimator.predict(xtest), ytest)
    print('[SVM] training: ', 'f1:', f1, 'kappa:', kappa, "done in %0.3fs." % (time() - t0))
    with open(batch + '_svm_clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(batch + '_svm_searcher.pickle',
              'wb') as sf:
        pickle.dump(searcher, sf)
    with open(batch + '_svm_estimator.pickle',
      'wb') as sfm:
        pickle.dump(estimator, sfm)
    return searcher, estimator, clf, f1, kappa

# %% [markdown]
# ### Logistic Regression

# %%
def lg_model(xtrain, xtest, ytrain, ytest, batch):
    t0 = time()
    param_grid = {'penalty': ['l1', 'l2', 'elasticnet'],
                  'max_iter': range(10, 50, 10)}
    clf = GridSearchCV(
                        LogisticRegression(multi_class='auto', n_jobs=-1), 
                        param_grid, 
                        scoring=scoring,
                        n_jobs=-1, 
                        cv=10
                      )
    searcher = clf.fit(xtrain, ytrain)
    estimator = searcher.best_estimator_
    f1, kappa = measures(estimator.predict(xtest), ytest)
    print('[LG] training: ', 'f1:', f1, 'kappa:', kappa, "done in %0.3fs." % (time() - t0))
    with open(batch + '_lg_clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(batch + '_lg_searcher.pickle',
              'wb') as sf:
        pickle.dump(searcher, sf)
    with open(batch + '_lg_estimator.pickle',
      'wb') as sfm:
        pickle.dump(estimator, sfm)
    return searcher, estimator, clf, f1, kappa

# %% [markdown]
# ### Decision Tree

# %%
def dt_model(xtrain, xtest, ytrain, ytest, batch):
    t0 = time()
    param_grid = {'min_samples_split': range(2, 403, 20)}
    clf = GridSearchCV(
        tree.DecisionTreeClassifier(), param_grid, scoring=scoring,
                        n_jobs=-1, 
                        cv=10
    )
    searcher = clf.fit(xtrain, ytrain)
    estimator = searcher.best_estimator_
    f1, kappa = measures(estimator.predict(xtest), ytest)
    print('[dt] training: ', 'f1:', f1, 'kappa:', kappa, "done in %0.3fs." % (time() - t0))
    with open(batch + '_dt_clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(batch + '_dt_searcher.pickle',
              'wb') as sf:
        pickle.dump(searcher, sf)
    with open(batch + '_dt_estimator.pickle',
      'wb') as sfm:
        pickle.dump(estimator, sfm)
    return searcher, estimator, clf, f1, kappa
# %% [markdown]
# ### Naive Bayes

# %%
def nb_model(xtrain, xtest, ytrain, ytest, batch):
    t0 = time()
    param_grid = {}
    clf = GridSearchCV(GaussianNB(), param_grid,
                        n_jobs=-1, 
                        cv=10)
    searcher = clf.fit(xtrain, ytrain)
    estimator = searcher.best_estimator_
    f1, kappa = measures(estimator.predict(xtest), ytest)
    print('[NB] training: ', 'f1:', f1, 'kappa:', kappa, "done in %0.3fs." % (time() - t0))
    with open(batch + '_nb_clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(batch + '_nb_searcher.pickle',
              'wb') as sf:
        pickle.dump(searcher, sf)
    with open(batch + '_nb_estimator.pickle',
      'wb') as sfm:
        pickle.dump(estimator, sfm)
    return searcher, estimator, clf, f1, kappa

def nn_model(xtrain, xtest, ytrain, ytest, batch):
    t0 = time()
    hidden_layer_sizes = (10, 50, 50, 10)
    param_grid = {
                  'alpha': [1e-5, 1e-4, 1e-3],
                  'solver': ['lbfgs', 'sgd', 'adam'],
                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
                 }
    if batch == 'lyrics':
        hidden_layer_sizes=(100, 200, 100, 50)
    clf = GridSearchCV(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=905, early_stopping=True), param_grid,
                        n_jobs=-1, 
                        cv=10)
    searcher = clf.fit(xtrain, ytrain)
    estimator = searcher.best_estimator_
    f1, kappa = measures(estimator.predict(xtest), ytest)
    print('[nn] training: ', 'f1:', f1, 'kappa:', kappa, "done in %0.3fs." % (time() - t0))
    with open(batch + '_nn_clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open(batch + '_nn_searcher.pickle',
              'wb') as sf:
        pickle.dump(searcher, sf)
    with open(batch + '_nn_estimator.pickle',
      'wb') as sfm:
        pickle.dump(estimator, sfm)
    return searcher, estimator, clf, f1, kappa



# %% [markdown]
# ## Data spliting

# %%
data_cls['genre'].value_counts()


# %%
target = {}
count = 1
for i in set(data_cls['genre'].tolist()):
    target[i] = count
    count += 1


# %%
target


# %%
data_cls['genre_'] = data_cls['genre']

data_cls['genre'] = data_cls['genre'].map(target)


# %%
# data_cls.columns

# %% [markdown]
# # Training
# %% [markdown]
# ## Emotion & Topic

# %%
X = data_cls[['clustered_label','Fear', 'Sad', 'Bored', 'Happy', 'Excited', 'Angry']].values
Y = data_cls['genre'].astype(int).values

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=31)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)


# %%
nn_searcher, nn_estimator, nn_clf, nn_f1, nn_kappa = nn_model(xtrain, xtest, ytrain, ytest, batch='Combined')
svm_searcher, svm_estimator, svm_clf, svm_f1, svm_kappa = svm_model(xtrain, xtest, ytrain, ytest, batch='Combined')
lg_searcher, lg_estimator, lg_clf, lg_f1, lg_kappa = lg_model(xtrain, xtest, ytrain, ytest, batch='Combined')
dt_searcher, dt_estimator, dt_clf, dt_f1, dt_kappa = dt_model(xtrain, xtest, ytrain, ytest, batch='Combined')
nb_searcher, nb_estimator, nb_clf, nb_f1, nb_kappa = nb_model(xtrain, xtest, ytrain, ytest, batch='Combined')

# %% [markdown]
# ### Lyrics

# %%
X_ = data
Y_ = data_cls['genre'].astype(int).values

xtrain_, xtest_, ytrain_, ytest_ = train_test_split(X_, Y_, test_size=0.2, random_state=31)
xtrain_.shape, xtest_.shape, ytrain_.shape, ytest_.shape


# %%
nn_searcher_, nn_estimator_, nn_clf_, nn_f1_, nn_kappa_ = nn_model(xtrain_, xtest_, ytrain_, ytest_, batch='lyrics')
svm_searcher_, svm_estimator_, svm_clf_, svm_f1_, svm_kappa_ = svm_model(xtrain_, xtest_, ytrain_, ytest_, batch='lyrics')
lg_searcher_, lg_estimator_, lg_clf_, lg_f1_, lg_kappa_ = lg_model(xtrain_, xtest_, ytrain_, ytest_, batch='lyrics')
dt_searcher_, dt_estimator_, dt_clf_, dt_f1_, dt_kappa_ = dt_model(xtrain_, xtest_, ytrain_, ytest_, batch='lyrics')
nb_searcher_, nb_estimator_, nb_clf_, nb_f1_, nb_kappa_ = nb_model(xtrain_, xtest_, ytrain_, ytest_, batch='lyrics')


# %%


