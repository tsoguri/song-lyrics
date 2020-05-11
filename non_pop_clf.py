# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:31:36 2020

@author: Eliot Shekhtman
"""

import numpy as np
import math, string, random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from subsample_trainer import SGDSubsampleClassifier
from subsample_trainer import CNBSubsampleClassifier
from subsample_trainer import CNBTwoStepClassifier
from subsample_trainer import SGDTwoStepClassifier
from subsample_trainer import RFTwoStepClassifier


classifiers = {'CNB' : ComplementNB(), 
               'SVM' : SGDClassifier(), 
               'RF' : RandomForestClassifier(max_depth=1000),
               'KNN' : KNeighborsClassifier(1),
               'SGDSC' : SGDSubsampleClassifier(),
               'CNBSC' : CNBSubsampleClassifier(),
               'CNB2C' : CNBTwoStepClassifier(),
               'SGD2C' : SGDTwoStepClassifier(),
               'RF2C' : RFTwoStepClassifier(max_depth=1000)} # 
    

def score_results(name, pipeline):
    print('>>> ' + name + ' classifier train/test scores: ')
    print('  n-p train: ' + str(pipeline.score(X_train_n, y_train_n)))
    print('      test:  '+str(pipeline.score(X_test_n, y_test_n)))
    print('  pop train: ' + str(pipeline.score(X_train_p, y_train_p)))
    print('      test:  '+str(pipeline.score(X_test_p, y_test_p)))
    print('  all train: ' + str(pipeline.score(X_train, y_train)))
    print('      test:  '+str(pipeline.score(X_test, y_test)))

def train_all(X_train, X_test, y_train, y_test):
    for name in classifiers:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifiers[name]),
            ])
        pipeline.fit(X_train, y_train)
        score_results(name, pipeline)

data_options = ['lyrics_spotify.csv', 'cleaned_scraped_lyrics_genre.csv']
data_choice = 'lyrics_spotify.csv'

if data_choice == 'lyrics_spotify.csv':

    data = pd.read_csv("lyrics_spotify.csv")
    
    dirty_dict = {"lyrics" : [],
                    "genre" : [],
                    "popularity" : []}
    # Basic cleaning of impromper imports
    for index in range(0, len(data['Lyrics'])):
        # print("debug")
        # print(data['Lyrics'][index])
        if index % 100 == 0:
            print('Checking song #'+str(index)+'/'+str(len(data['Lyrics'])))
        if(isinstance(data['Lyrics'][index], str)):
            dirty_dict['lyrics'].append(data['Lyrics'][index])
            dirty_dict['genre'].append(data['Genre-group'][index])
            dirty_dict['popularity'].append(data['Popularity'][index])
    
    df_dirty = pd.DataFrame(dirty_dict)
    # no_pop_df_dirty = df_dirty[df_dirty['genre'] != 'pop']
    # pop_df_dirty = df_dirty[df_dirty['genre'] == 'pop']
    
    df = df_dirty
    # no_pop_df = no_pop_df_dirty
    # pop_df = pop_df_dirty
elif data_choice == 'cleaned_scraped_lyrics_genre.csv':
    df = pd.read_csv('cleaned_scraped_lyrics_genre.csv') # trigram_lyrics
    # no_pop_df = df[df['genre'] != 'pop']
    # pop_df = df[df['genre'] == 'pop']
else:
    raise ValueError('Improper CSV')



# Classify
print('Training classifiers')
X_train, X_test, y_train, y_test = train_test_split(df['lyrics'], df['genre'], test_size=0.3, random_state=0) # random_state=0
X_train_n, y_train_n = list(zip(*[(s,g) for s,g in list(zip(X_train,y_train)) if g != 'pop']))
X_test_n, y_test_n = list(zip(*[(s,g) for s,g in list(zip(X_test,y_test)) if g != 'pop']))
X_train_p, y_train_p = list(zip(*[(s,g) for s,g in list(zip(X_train,y_train)) if g == 'pop']))
X_test_p, y_test_p = list(zip(*[(s,g) for s,g in list(zip(X_test,y_test)) if g == 'pop']))
# X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(no_pop_df['lyrics'], no_pop_df['genre'], test_size=0.5, random_state=0)
# X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(pop_df['lyrics'], pop_df['genre'], test_size=0.5, random_state=0)

print((len(X_test_p)+len(X_train_p))/(len(X_test)+len(X_train)))



# X_test_skew, y_test_skew = remove_pop(X_test, y_test)

train_all(X_train, X_test, y_train, y_test)
# print('########## Training solely on non-pop ###########')
# train_all(X_train_n, X_test_n, y_train_n, y_test_n)





# text_cnb_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', ComplementNB()),
#     ])
# text_cnb_clf.fit(X_train, y_train)
# score_results('CNB', text_cnb_clf)

# text_sgd_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier()),
#     ])
# text_sgd_clf.fit(X_train, y_train)
# score_results('SVM', text_sgd_clf)

# text_rf_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', RandomForestClassifier()),
#     ])
# text_rf_clf.fit(X_train, y_train)
# score_results('RF', text_rf_clf)

# print('########## Training solely on non-pop ###########')
# text_cnb_clf_np = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', ComplementNB()),
#     ])
# text_cnb_clf_np.fit(X_train_n, y_train_n)
# score_results('CNB', text_cnb_clf_np)

# text_sgd_clf_np = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier()),
#     ])
# text_sgd_clf_np.fit(X_train_n, y_train_n)
# score_results('SVM', text_sgd_clf_np)

# text_rf_clf_np = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', RandomForestClassifier()),
#     ])
# text_rf_clf_np.fit(X_train_n, y_train_n)
# score_results('RF', text_rf_clf_np)