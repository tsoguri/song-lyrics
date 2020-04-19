# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:44:14 2020

@author: Eliot Shekhtman
"""

import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

def findw(w1, w2):
    w1l = re.split(r"\W+", w1)
    return w2 in w1l

def group_genre(top_genre):
    top_genre = ' '+top_genre+' '
    if findw(top_genre, 'hip'):
        return 'hip hop'
    elif findw(top_genre, 'pop'):
        return 'pop'
    elif findw(top_genre, 'soul'):
        return 'r&b'
    elif findw(top_genre, 'r') and findw(top_genre, 'b'):
        return 'r&b'
    elif findw(top_genre, 'boy'):
        return 'boy band'
    elif findw(top_genre, 'rap'):
        return 'rap'
    elif findw(top_genre, 'rock'):
        return 'rock'
    elif findw(top_genre, 'room'):
        return 'house'
    elif findw(top_genre, 'house'):
        return 'house'
    elif findw(top_genre, 'metropopolis'):
        return 'pop'
    elif findw(top_genre, 'indie'):
        return 'indie'
    elif findw(top_genre, 'singer'):
        return 'indie'
    elif findw(top_genre, 'techno'):
        return 'electronic'
    elif findw(top_genre, 'edm'):
        return 'electronic'
    elif findw(top_genre, 'complextro'):
        return 'electronic'
    elif findw(top_genre, 'electro'):
        return 'electronic'
    elif findw(top_genre, 'electropop'):
        return 'electronic'
    elif findw(top_genre, 'wave'):
        return 'alternative'
    elif findw(top_genre, 'brostep'):
        return 'electronic'
    elif findw(top_genre, 'dubstep'):
        return 'electronic'
    elif findw(top_genre, 'dance'):
        return 'pop'
    elif findw(top_genre, 'latin'):
        return 'latin'
    else:
        return top_genre.strip()

def infer_metrolyrics_url(title, artist):
    delimiter_re = r"(\(.*\))*\W"
    url = 'https://www.metrolyrics.com/'
    title = [w for w in re.split(delimiter_re, title.lower()) if w != '' and w is not None and w[0] != '(']
    artist = [w for w in re.split(delimiter_re, artist.lower()) if w != '' and w is not None and w[0] != '(']
    for word in title:
        url += word + '-'
    url += 'lyrics'
    for word in artist:
        url += '-' + word
    url += '.html'
    return url
    

song_dataset = pd.read_csv('top10s.csv')
lyrics_dataset = {'Song': [],
                  'Artist': [],
                  'Year': [],
                  'Genre-group': [],
                  'Lyrics': []}

counter = 0
for index in range(0, len(song_dataset['title'])):
    if index % 100 == 0:
        print('Reading song #'+str(index)+'/'+str(len(song_dataset['title']))+': '+str(counter)+' successful')
    title = song_dataset['title'][index]
    artist = song_dataset['artist'][index]
    top_genre = song_dataset['top genre'][index]
    year = song_dataset['year'][index]
    URL = infer_metrolyrics_url(title, artist)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    lyricbody = soup.find(id='lyrics-body')
    if lyricbody is None: # Means that it pulled up a 404 page most likely
        continue
    verses_iterable = lyricbody.find_all('p', class_='verse')
    lyrics = ''
    for verse in verses_iterable:
        lyrics += verse.text.strip() + ' '
    lyrics_dataset['Song'].append(title)
    lyrics_dataset['Artist'].append(artist)
    lyrics_dataset['Year'].append(year)
    lyrics_dataset['Genre-group'].append(group_genre(top_genre))
    lyrics_dataset['Lyrics'].append(lyrics)
    counter += 1

print('Processing done! Saving data...')
df = pd.DataFrame(lyrics_dataset)
df.to_csv('lyrics_spotify.csv', encoding='utf-8')
print('Done!')
        

# metro lyrics: title-lyrics-author, parts in parentheses omitted, all
# lowercase, dashes between everything

# URL = 'https://www.metrolyrics.com/hey-soul-sister-lyrics-train.html'
# page = requests.get(URL)
# print(page.text)

# soup = BeautifulSoup(page.content, 'html.parser')
# results_lyricbody = soup.find(id='lyrics-body')
# print(results_lyricbody.prettify())
# verses_iterable = results_lyricbody.find_all('p', class_='verse')
# for verse in verses_iterable:
#     print(verse.text.strip(), end='\n'*2)

