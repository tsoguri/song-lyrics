# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:31:43 2020

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
from sklearn.mixture import BayesianGaussianMixture
import operator
from abc import ABC, abstractmethod


class ImbalancedTrainerInterface(ABC):
    def __init__(self):
        super().__init__()
        
    
    def _find_dominant_class(self, x, y): # TODO: make more comprehensive, deal with multiple dom classes
        classes = dict(zip(set(y), [0]*len(set(y))))
        for e in y:
            classes[e] += 1
        class_ratios = [(k, v / len(y)) for k, v in classes.items()] 
        max_class = max(class_ratios, key=lambda x : x[1])
        # print(max_class)
        return max_class
    
    
    def _partition(self, x, y, dom):
        xy = list(zip(x, y))
        sub_with = [(xe, ye) for xe,ye in xy if ye == dom]
        sub_without = [(xe, ye) for xe,ye in xy if ye != dom]
        x_with, y_with = list(zip(*sub_with))
        x_without, y_without = list(zip(*sub_without))
        return list(x_with), list(x_without), list(y_with), list(y_without)



class SubsampleTrainer(ImbalancedTrainerInterface):
    def __init__(self, X_train, y_train, clf_type=None, arg=None):
        # X_train = list(X_train)
        y_train = pd.Series(y_train)
        # later make this so it only takes in X_train and y_train and finds out which are way too large and splits it
        self.type = clf_type
        self.__x_subs, self.__y_subs = self.__subset(X_train, y_train)
        self.__pipelines = None
        self.classes = np.unique(y_train)
        self.extra_arg = arg

    
    def __subset(self, x, y, num_subsets = 10, num_per_subset = 20):
        max_class = self._find_dominant_class(x, y)
        # print(x)
        x_w, x_o, y_w, y_o = self._partition(x, y, max_class[0])
        x_subs = [None] * num_subsets
        y_subs = [None] * num_subsets
        for i in range(num_subsets):
            x_subs[i] = list(np.array(x_o).copy())
            y_subs[i] = list(np.array(y_o).copy())
        # x_subs = [x_o] * num_subsets
        # y_subs = [y_o] * num_subsets
        # print(y_subs[0])
        for i in range(num_subsets):
            num_pick = num_per_subset if num_per_subset < len(x_w) else len(x_w)-1
            sample = random.sample(list(zip(x_w, y_w)), num_pick)
            # print(num_pick)
            # print(list(sample))
            if(list(sample) == []):
                x_sample = []
                y_sample = []
            else:
                x_sample, y_sample = list(zip(*sample))
            # print(np.array(x_sample).shape)
            x_subs[i] += list(x_sample)
            y_subs[i] += list(y_sample)
        # print(x_subs)
        # print(y_subs[0])
        # print(y_o)
        return x_subs, y_subs
    
    
    def fit(self, clf_type=None):
        if clf_type == None and self.type == None:
            raise ValueError('Must give a classifier to fit with')
        if clf_type != None:
            self.type = clf_type
        num_subsets = len(self.__x_subs)
        clfs = [None] * num_subsets
        for i in range(num_subsets):
            if(self.extra_arg == None):
                clf = self.type()
            else:
                clf = self.type(self.extra_arg)
            # print(clf)
            # print(self.__x_subs[i])
            clf.fit(self.__x_subs[i], self.__y_subs[i])
            clfs[i] = clf
        self.classifiers = clfs
        
    
    def predict(self, X_test):
        # X_test = list(X_test)
        num_subsets = len(self.__x_subs)
        y_preds = []
        for clf in self.classifiers:
            y_preds.append(clf.predict_proba(X_test))
        # y_preds = [clf.predict_proba(X_test) for clf in self.classifiers]
        
        # y_preds = np.array(y_preds).transpose(1,0,2)
        # print(y_preds)
        # print(y_preds.shape)
        # avg_predictions = [np.mean(y_of_xs) for y_of_xs in y_preds]
        y_preds = np.array(y_preds).transpose(1,0,2)
        y_labels = []
        for song in y_preds:
            genre_probs = {}
            for c in self.classes:
                genre_probs[c] = 0
            for clf_res, clf in list(zip(song, self.classifiers)):
                for prob, genre in list(zip(clf_res, clf.classes_)):
                    genre_probs[genre] += prob
            most_likely = max(genre_probs.items(), key=operator.itemgetter(1))[0]
            y_labels.append(most_likely)
        
        return y_labels
    
    
    def score(self, X_test, y_test):
        # X_test = list(X_test)
        y_test = pd.Series(y_test)
        y_pred = self.predict(X_test)
        # print(X_test.shape[0])
        # print(len(y_test))
        # print(len(y_pred))
        acc = np.mean(y_pred == y_test)
        return acc

class SGDSubsampleClassifier(SubsampleTrainer):
    def __init__(self):
        self.type = SGDClassifier
    
    def fit(self, X_train, y_train):
        # print(X_train)
        # print(type(X_train))
        # X_train_c = X_train.tocoo()
        # X_train_ll = [X_train_c.col.tolist(), X_train_c.data.tolist()]
        SubsampleTrainer.__init__(self, X_train.toarray().tolist(), y_train, self.type, 'log')
        SubsampleTrainer.fit(self, SGDClassifier)


class CNBSubsampleClassifier(SubsampleTrainer):
    def __init__(self):
        self.type = ComplementNB
        
    def fit(self, X_train, y_train):
        SubsampleTrainer.__init__(self, X_train.toarray().tolist(), y_train, ComplementNB)
        SubsampleTrainer.fit(self, ComplementNB)

class CNBTwoStepClassifier(ImbalancedTrainerInterface):
    def __init__(self):
        self.clf_yn = ComplementNB()
        self.clf_n = ComplementNB()
    
    def fit(self, X_train, y_train):
        X_train = X_train.toarray().tolist()
        y_train = pd.Series(y_train)
        max_class = self._find_dominant_class(X_train, y_train)
        x_w, x_o, y_w, y_o = self._partition(X_train, y_train, max_class[0])
        x_yn = x_w + x_o
        y_yn = y_w + ['not'] * len(y_o)
        # print(y_yn)
        # print(y_o)
        self.clf_yn.fit(x_yn, y_yn)
        self.clf_n.fit(x_o, y_o)
    
    def predict(self, X_test):
        y_pred_yn = self.clf_yn.predict(X_test)
        y_pred_total = []
        for p,x in list(zip(y_pred_yn, X_test)):
            if p == 'not':
                y_pred_total.append(self.clf_n.predict(x)[0])
            else:
                y_pred_total.append(p)
        return y_pred_total
    
    def score(self, X_test, y_test):
        y_test = pd.Series(y_test)
        y_pred = self.predict(X_test)
        acc = np.mean(y_pred == y_test)
        return acc

class SGDTwoStepClassifier(ImbalancedTrainerInterface):
    def __init__(self):
        self.clf_yn = SGDClassifier()
        self.clf_n = SGDClassifier()
    
    def fit(self, X_train, y_train):
        X_train = X_train.toarray().tolist()
        y_train = pd.Series(y_train)
        max_class = self._find_dominant_class(X_train, y_train)
        x_w, x_o, y_w, y_o = self._partition(X_train, y_train, max_class[0])
        x_yn = x_w + x_o
        y_yn = y_w + ['not'] * len(y_o)
        # print(x_yn)
        # print(y_o)
        self.clf_yn.fit(x_yn, y_yn)
        self.clf_n.fit(x_o, y_o)
    
    def predict(self, X_test):
        y_pred_yn = self.clf_yn.predict(X_test)
        y_pred_total = []
        for p,x in list(zip(y_pred_yn, X_test)):
            if p == 'not':
                y_pred_total.append(self.clf_n.predict(x)[0])
            else:
                y_pred_total.append(p)
        return y_pred_total
    
    def score(self, X_test, y_test):
        y_test = pd.Series(y_test)
        y_pred = self.predict(X_test)
        acc = np.mean(y_pred == y_test)
        return acc


class RFTwoStepClassifier(ImbalancedTrainerInterface):
    def __init__(self, max_depth=None):
        self.clf_yn = RandomForestClassifier(max_depth=max_depth)
        self.clf_n = RandomForestClassifier(max_depth=max_depth)
    
    def fit(self, X_train, y_train):
        X_train = X_train.toarray().tolist()
        y_train = pd.Series(y_train)
        max_class = self._find_dominant_class(X_train, y_train)
        x_w, x_o, y_w, y_o = self._partition(X_train, y_train, max_class[0])
        x_yn = x_w + x_o
        y_yn = y_w + ['not'] * len(y_o)
        # print(x_yn)
        # print(y_o)
        self.clf_yn.fit(x_yn, y_yn)
        self.clf_n.fit(x_o, y_o)
    
    def predict(self, X_test):
        y_pred_yn = self.clf_yn.predict(X_test)
        y_pred_total = []
        for p,x in list(zip(y_pred_yn, X_test)):
            if p == 'not':
                y_pred_total.append(self.clf_n.predict(x)[0])
            else:
                y_pred_total.append(p)
        return y_pred_total
    
    def score(self, X_test, y_test):
        y_test = pd.Series(y_test)
        y_pred = self.predict(X_test)
        acc = np.mean(y_pred == y_test)
        return acc

