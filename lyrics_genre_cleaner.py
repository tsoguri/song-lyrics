# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:31:36 2020

@author: Eliot Shekhtman
"""

import numpy as np
import math, string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

remove_stopwords = False

def fix_unnecessary_tokenization(wordlist):
    newlist = []
    i = 0
    while i < len(wordlist)-1:
        if wordlist[i] == 'got' and wordlist[i+1] == 'ta':
            newlist.append('gotta')
            i += 2
        elif wordlist[i] == 'gon' and wordlist[i+1] == 'na':
            newlist.append('gonna')
            i += 2
        elif wordlist[i] == 'wan' and wordlist[i+1] == 'na':
            newlist.append('wanna')
            i += 2
        else:
            newlist.append(wordlist[i])
            i += 1
    if i < len(wordlist):
        newlist.append(wordlist[i])
    return newlist


data = pd.read_csv("lyrics_spotify.csv")
columns = data.columns
stop_words = set(stopwords.words('english') + list(string.punctuation))
stop_words_ext = set(stopwords.words('english') + list(string.punctuation) + ['yeah', 'oh', 'ohh', 'woah', 'la', 'na'])
ps = PorterStemmer()
# Create dummy lists to populate with values
lyrics_tokens = []
genre_targets = []
popularity_targets = []

genre_counts = {}
genre_words = {}
all_words = []

# Break lyrics up into tokens and get genre counts
for index in range(0, len(data['Lyrics'])):
    # print("debug")
    # print(data['Lyrics'][index])
    if index % 100 == 0:
        print('Tokenizing song #'+str(index)+'/'+str(len(data['Lyrics'])))
    if data['Genre-group'][index] in genre_counts:
        genre_counts[data['Genre-group'][index]] += 1
    else:
        genre_counts[data['Genre-group'][index]] = 1
        genre_words[data['Genre-group'][index]] = []
    if(isinstance(data['Lyrics'][index], str)):
        lyrics = (data['Lyrics'][index]).translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        lyrics_tokenized = [w.lower() for w in word_tokenize(lyrics) if not w in stop_words_ext] # clean stopwords
        lyrics_tokenized = fix_unnecessary_tokenization(lyrics_tokenized)
        lyrics_tokens.append(lyrics_tokenized) # Get song data per song
        genre_words[data['Genre-group'][index]] += lyrics_tokenized
        all_words += lyrics_tokenized
        genre_targets.append(data['Genre-group'][index]) # Get targets
        popularity_targets.append(data['Popularity'][index])
unique_words = list(dict.fromkeys(all_words))
# print(genre_counts)
print(len(lyrics_tokens))

# Now treat words that occur fewer than 10 times across all classes as unknown words (disregard)
known_words = []
print('Removing outlier words')
for index in range(0, len(unique_words)):
    w = unique_words[index]
    if index % 100 == 0:
        print('Checking word #'+str(index)+'/'+str(len(unique_words)))
    known = False
    for g in genre_words:
        if genre_words[g].count(w) >= 10:
            known = True
    if known:
        known_words.append(w)
    else:
        for g in genre_words:
            genre_words[g] = [r for r in genre_words[g] if r != w]
        for i in range(0, len(lyrics_tokens)):
            lyrics_tokens[i] = [r for r in lyrics_tokens[i] if r != w]
print(len(lyrics_tokens))

# Re-concatenate into cleaned lyrics for sklearn
lyrics = ['']*len(lyrics_tokens)
for i in range(0, len(lyrics_tokens)):
    if i % 100 == 0:
        print('Reforming song #'+str(i)+'/'+str(len(lyrics_tokens)))
    for w in lyrics_tokens[i]:
        lyrics[i] += w + ' '
print(len(lyrics_tokens))

total_data = {'lyrics' : lyrics, 'genre' : genre_targets, 'popularity' : popularity_targets}
df = pd.DataFrame(total_data)
df.to_csv('cleaned_scraped_lyrics_genre.csv', encoding='utf-8')

