## What Can Popular Music Tell Us? 

# Song Lyrics Data Visualization and Analysis

Final Report: https://docs.google.com/document/d/1VhnYw_w_kScZHeUjz3jmKTcn5msV56YkaG90i-fJrKI/edit?usp=sharing

"Is that song by ___ ? It sounds like them."
"I don’t like ___ , it all sounds the same to me."
"Oh this song is going to be a hit."

We often hear these comments about the music we listen to, as we unknowingly create correlations about certain artists, genres, and types of popular songs in our heads. But, are these correlations actually rooted in something quantifiable? Are we able to make predictions based on these patterns and correlations? As avid music lovers, we wanted to learn more about the possible patterns in the songs we listen to. We chose to analyze a Spotify top tracks dataset and took a look at various features associated with the top 50 tracks of each year from 2010 to 2019, which included values for beats per minute, popularity, loudness, etc. This dataset also included standard information such as artist, year, and genre. We were also interested in the actual contents of songs, and thus also found a Billboard top songs dataset that had lyrics of over 5000 popular songs. We combined these datasets by matching on the titles and artist with a Python script (in the linked Github repo). We used Python Pandas, collections, NLTK, SKLearn, BeautifulSoup, and Matplotlib libraries as well as Tableau. 


