# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:06:50 2020

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


classifiers = {'CNB' : ComplementNB, 
               'SVM' : SGDClassifier, 
               'RF' : RandomForestClassifier,
               'KNN' : KNeighborsClassifier,
               } # 'SGDSC' : SGDSubsampleClassifier
    

def score_results(name, pipeline, X_train, X_test, y_train, y_test):
    print('>>> ' + name + ' classifier train/test scores: ')
    print('  train: ' + str(pipeline.score(X_train, y_train)))
    print('  test:  '+str(pipeline.score(X_test, y_test)))

def train_all(X_train, X_test, y_train, y_test):
    for name in classifiers:
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifiers[name]()),
            ])
        pipeline.fit(X_train, y_train)
        score_results(name, pipeline, X_train, X_test, y_train, y_test)

def discretize_popularity(df):
    replacements = {}
    for i in range(100):
        replacements[i] = i // 20
    df = df.replace({'popularity' : replacements})
    return df

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
df_dirty = discretize_popularity(df_dirty)

df = pd.read_csv('cleaned_scraped_lyrics_genre.csv')
df = discretize_popularity(df)



# Classify
print('Training classifiers')
X_train, X_test, y_train, y_test = train_test_split(df['lyrics'], df['popularity'], test_size=0.5, random_state=0) # random_state=0
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(df_dirty['lyrics'], df_dirty['popularity'], test_size=0.5, random_state=0)





# X_test_skew, y_test_skew = remove_pop(X_test, y_test)

train_all(X_train, X_test, y_train, y_test)
print('########## Training using dirty data ###########')
train_all(X_train_d, X_test_d, y_train_d, y_test_d)