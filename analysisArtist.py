import pandas as pd
import csv
from nltk.corpus import stopwords
import collections
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
stop_words.add("yeah")
stop_words.add("like")
stop_words.add("yeah")
stop_words.add("im")
stop_words.add("one")
stop_words.add("youre")
stop_words.add("gonna")
stop_words.add("got")
stop_words.add("dont")
stop_words.add("make")
stop_words.add("oh")
stop_words.add("let")
stop_words.add("")

df = pd.read_csv('new.csv')

for index, row in df.iterrows():
    df.at[index, 'Lyrics'] = row['Lyrics'].split(" ") if row['Lyrics'] else []

def word_count(name):
    s = df.loc[df['artist'] == name]
    count_songs = 0
    count_words = set()
    for index, row in s.iterrows():
        count_songs += 1
        for w in df.at[index, 'Lyrics']:
            count_words.add(w)
    if count_songs == 0:
        print(name)
    else:
        return len(count_words)/count_songs

top_artists = (df['artist'].value_counts()[:10].index.tolist())
top_artists = ['Rihanna', 'Katy Perry', 'Maroon 5', 'Nicki Minaj', 'Eminem', 'Taio Cruz', 'The Black Eyed Peas', 'Calvin Harris', 'Iggy Azalea', 'Pitbull']
top = {}
for art in top_artists:
    if art == "The Black Eyed Peas":
        top['BEP'] = word_count(art)
    else:
        top[art] = word_count(art)

plt.bar(*zip(*top.items()), color=['pink', '#FF9AA2', '#FFB7B2', '#FFDAC1', '#FCF7DE', '#E2F0CB', '#B5EAD7', '#DEF3FD', '#C7CEEA', '#F0DEFD'])
plt.xticks(rotation=75)
plt.xlabel('Top Artists')
plt.ylabel('Number of Average Unique Words per Song')

plt.show()


ri = df.loc[df['artist'] == 'Rihanna']
ri_list2 = []
for index, row in ri.iterrows():
    for w in df.at[index, 'Lyrics']:
        if w not in stop_words:
            ri_list2.append(w)
listToStr1 = ' '.join([str(elem) for elem in ri_list2])

kp = df.loc[df['artist'] == 'Pitbull']

kp_list2 = []
for index, row in kp.iterrows():
    for w in df.at[index, 'Lyrics']:
        if w not in stop_words:
            kp_list2.append(w)
listToStr2 = ' '.join([str(elem) for elem in kp_list2])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10).generate(listToStr2)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()

most_common_artist = (df["artist"].mode())

#df.to_csv("Search.csv", index=False, encoding='utf8')
