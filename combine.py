import pandas as pd
import csv
import re

spotify_df = pd.read_csv('songdata.csv', sep=',', engine='python')
billboard_df = pd.read_csv('billboard.csv', sep=',', engine='python')

spotify_df['clean'] = ""

for index, row in spotify_df.iterrows():
    val = (re.sub('[^a-zA-Z.\d\s]', '', row['title'].lower()))
    spotify_df.at[index, 'clean'] = val

billboard_df = billboard_df[billboard_df.Year > 2009]

merged = pd.merge(spotify_df, billboard_df, left_on='clean', right_on='Song', how='left')
merged = merged.drop_duplicates(subset = 'title', keep = 'first')
merged = merged.dropna()
merged = merged.drop(['clean', 'Rank', 'Song', 'Artist', 'Year', 'Source'], axis = 1)

merged.to_csv('new.csv')
