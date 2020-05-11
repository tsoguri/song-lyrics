# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:40:27 2020

@author: Eliot Shekhtman
"""

import math, string
import pandas as pd
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
# lyrics_tokens = [None]*len(data['Lyrics'])

song_dict = {'lyrics' : [],
             'genre' : [],
             'popularity' : []}

SEP = ''

# Break lyrics up into tokens
for index in range(0, len(data['Lyrics'])):
    # print("debug")
    # print(data['Lyrics'][index])
    if index % 100 == 0:
        print('Tokenizing song #'+str(index)+'/'+str(len(data['Lyrics'])))
    if(isinstance(data['Lyrics'][index], str)):
        lyrics = (data['Lyrics'][index]).translate(str.maketrans('', '', string.punctuation))
        tokens = [w.lower() for w in word_tokenize(lyrics) if not w in list(string.punctuation)]
        tokens = fix_unnecessary_tokenization(tokens)
        new_lyrics = ''
        for i in range(len(tokens)-2):
            trigram = tokens[i]+SEP+tokens[i+1]+SEP+tokens[i+2]
            new_lyrics += trigram + ' '
        song_dict['lyrics'].append(new_lyrics)
        song_dict['genre'].append(data['Genre-group'][index])
        song_dict['popularity'].append(data['Popularity'][index])

print('Processing done! Saving data...')
df = pd.DataFrame(song_dict)
df.to_csv('trigram_lyrics.csv', encoding='utf-8')
print('Done!')




