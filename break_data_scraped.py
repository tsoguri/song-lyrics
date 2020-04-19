# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:09:49 2020

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
lyrics_tokens = [None]*len(data['Lyrics'])

# Break lyrics up into tokens
for index in range(0, len(data['Lyrics'])):
    # print("debug")
    # print(data['Lyrics'][index])
    if index % 100 == 0:
        print('Tokenizing song #'+str(index)+'/'+str(len(data['Lyrics'])))
    if(isinstance(data['Lyrics'][index], str)):
        lyrics = (data['Lyrics'][index]).translate(str.maketrans('', '', string.punctuation))
        lyrics_tokens[index] = [w.lower() for w in word_tokenize(lyrics) if not w in list(string.punctuation)]
        lyrics_tokens[index] = fix_unnecessary_tokenization(lyrics_tokens[index])
    else:
        lyrics_tokens[index] = []
# Add column to dataframe
data['Tokens'] = lyrics_tokens

# Process Unigrams
unigram_dict = {'Unigram': [],
                'Song': [], 
                'Year': [],
                'Genre': []}

for i in range(0, len(data['Tokens'])):
    if i % 100 == 0:
        print('Unigramizing song #'+str(i)+'/'+str(len(data['Tokens'])))
    for j in range(0, len(data['Tokens'][i])):
        unigram = data['Tokens'][i][j]
        if remove_stopwords and unigram in stop_words_ext:
            continue
        unigram_dict['Unigram'].append(unigram)
        unigram_dict['Song'].append(data['Song'][i])
        unigram_dict['Year'].append(data['Year'][i])
        unigram_dict['Genre'].append(data['Genre-group'][i])

print('Processing done! Saving data...')
unigram_df = pd.DataFrame(unigram_dict)
unigram_df.to_csv('unigram_scraped.csv', encoding='utf-8')
print('Done!')

# Process Bigrams
bigram_dict = {'Bigram': [],
                'Song': [], 
                'Year': [],
                'Genre': []}

for i in range(0, len(data['Tokens'])):
    if i % 100 == 0:
        print('Bigramizing song #'+str(i)+'/'+str(len(data['Tokens'])))
    for j in range(0, len(data['Tokens'][i])-1):
        if remove_stopwords and ((data['Tokens'][i][j] in stop_words_ext) or (data['Tokens'][i][j+1] in stop_words_ext)):
            continue
        bigram = data['Tokens'][i][j]+' '+data['Tokens'][i][j+1]
        bigram_dict['Bigram'].append(bigram)
        bigram_dict['Song'].append(data['Song'][i])
        bigram_dict['Year'].append(data['Year'][i])
        bigram_dict['Genre'].append(data['Genre-group'][i])

print('Processing done! Saving data...')
bigram_df = pd.DataFrame(bigram_dict)
bigram_df.to_csv('bigram_scraped.csv', encoding='utf-8')
print('Done!')

# Process Trigrams
# For now, for visualization purposes just need 3 words in a row, don't need anything fancy
trigram_dict = {'Trigram': [],
                'Song': [], 
                'Year': [],
                'Genre': []}

for i in range(0, len(data['Tokens'])):
    if i % 100 == 0:
        print('Trigramizing song #'+str(i)+'/'+str(len(data['Tokens'])))
    for j in range(0, len(data['Tokens'][i])-2):
        if remove_stopwords and ((data['Tokens'][i][j] in stop_words_ext) or (data['Tokens'][i][j+1] in stop_words_ext) or (data['Tokens'][i][j+2] in stop_words_ext)):
            continue
        trigram = data['Tokens'][i][j]+' '+data['Tokens'][i][j+1]+' '+data['Tokens'][i][j+2]
        trigram_dict['Trigram'].append(trigram)
        trigram_dict['Song'].append(data['Song'][i])
        trigram_dict['Year'].append(data['Year'][i])
        trigram_dict['Genre'].append(data['Genre-group'][i])

print('Processing done! Saving data...')
trigram_df = pd.DataFrame(trigram_dict)
trigram_df.to_csv('trigrams_scraped.csv', encoding='utf-8')
print('Done!')
