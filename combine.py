import pandas as pd
import csv
import re

spotify_df = pd.read_csv('songdata.csv', sep=',', engine='python')


spotify_df['clean'] = ""

for index, row in spotify_df.iterrows():
    val = (re.sub('[^a-zA-Z.\d\s]', '', row['title'].lower()))
    spotify_df.at[index, 'clean'] = val
spotify_df.to_csv('new.csv')
