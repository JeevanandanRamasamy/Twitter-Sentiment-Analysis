import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
sns.set_context("talk")
plt.style.use('fivethirtyeight')

import nltk
import nltk.corpus
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

def load_tweets(path):
    """Loads tweets that have previously been saved.
    
    Calling load_tweets(path) after save_tweets(tweets, path)
    will produce the same list of tweets.
    
    Args:
        path (str): The place where the tweets will be saved.

    Returns:
        list: A list of Dictionary objects, each representing one tweet."""
    
    with open(path, "rb") as f:
        import json
        return json.load(f)

# Merge the two csv files into one DataFrame
trump_tweets1 = pd.DataFrame(load_tweets('TrumpTweets_1.json'))
trump_tweets2 = pd.DataFrame(load_tweets('TrumpTweets_2.json'))
trump_tweets1.columns = ['source', 'id', 'full_text', 'created_at', 'retweet_count', 'in_reply_to_user_id', 'favorite_count', 'retweeted']
trump_tweets1['id'] = trump_tweets1['id'].astype(int)
trump_tweets1['in_reply_to_user_id'] = trump_tweets1['in_reply_to_user_id'].astype(float)
all_tweets = pd.merge(trump_tweets1, trump_tweets2, on=['source', 'id', 'full_text', 'created_at', 'retweet_count', 'in_reply_to_user_id', 'favorite_count', 'retweeted'], how='outer')
print(all_tweets.head())

# Data Cleaning
df_trump = all_tweets.drop_duplicates(subset='id')
df_trump.set_index('id', inplace=True)
df_trump = df_trump[['created_at', 'source', 'full_text', 'retweet_count', 'favorite_count']]
df_trump.columns = ['time', 'source', 'text', 'retweet_count', 'favorite_count']
df_trump['time'] = pd.to_datetime(df_trump['time'], format='%a %b %d %H:%M:%S +0000 %Y')
df_trump = df_trump.sort_values(by='time')

# Remove HTML tags from the source column
print(df_trump['source'].unique())
df_trump['source'] = df_trump['source'].str.replace(r'<[^>]*>', '', regex=True)
df_trump['source'].value_counts().plot(kind='bar')
plt.title('Source of Trump Tweets')
plt.ylabel('Number of Tweets')
plt.xlabel('Source')
plt.show()

# Convert to Eastern Time
df_trump['est_time'] = (
    df_trump['time'].dt.tz_localize("UTC")
                 .dt.tz_convert("EST") # Convert to Eastern Time
)
print(df_trump.head())

# Visualize hourly tweet frequency
df_trump['hour'] = df_trump['est_time'].dt.hour + df_trump['est_time'].dt.minute/60 + df_trump['est_time'].dt.second/3600
df_trump['roundhour']=round(df_trump['hour'])
df_trump['roundhour'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Trump Tweets by Hour')
plt.ylabel('Number of Tweets')
plt.xlabel('Hour')
plt.show()

# Compare tweet frequency by source
def plot_kde(df):
    iphone = df[df['source'] == 'Twitter for iPhone']['roundhour']
    android = df[df['source'] == 'Twitter for Android']['roundhour']
    sns.distplot(iphone, hist=False, label='iPhone')
    sns.distplot(android, hist=False, label='Android')
    plt.ylabel('Fraction')
    plt.xlabel('Hour')
    plt.legend()
    plt.show()
plot_kde(df_trump)

# Compare tweet frequency by source in 2016 (election year)
df_trump_2016 = df_trump[df_trump['time'].dt.year == 2016]
plot_kde(df_trump_2016)

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length
df_trump['year'] = df_trump['time'].apply(year_fraction)

# Plot KDE of tweet frequency by source over time
def plot_kde_hist(df):
    plt.figure(figsize=(15,15))
    iphone = df[df['source'] == 'Twitter for iPhone']['year']
    android = df[df['source'] == 'Twitter for Android']['year']
    sns.distplot(iphone, label='iPhone')
    sns.distplot(android, label='Android')
    plt.ylabel('Fraction')
    plt.xlabel('Year')
    plt.legend()
    plt.show()
plot_kde_hist(df_trump)

# Read in VADER sentiment lexicon
with open("vader_lexicon.txt") as f:
    df_sent = pd.read_csv(f, sep='\t', names=['token', 'polarity', 'sd', 'raw'])
    df_sent = df_sent.drop(['sd', 'raw'], axis=1)
    df_sent = df_sent.groupby('token').mean()

# Calculate sentiment of each tweet
def get_sentiment(text):
    sentiment = 0
    words = text.split()
    for word in words:
        if word in df_sent.index:
            sentiment += df_sent.loc[word, 'polarity']
    return sentiment
    
df_trump['text'] = df_trump['text'].str.lower()
df_trump['sentiment'] = df_trump['text'].apply(get_sentiment)

# Remove punctuation
punct_re = r'[^\w\s\\n]'
df_trump['no_punc'] = df_trump['text'].str.replace(punct_re, ' ', regex=True)

# Split into words
tidy_format = df_trump['no_punc'].str.split(expand=True).stack().reset_index(level=1)
tidy_format.columns = ['num', 'word']

# Merge with sentiment data
merged_df = tidy_format.merge(df_sent, how='left', left_on='word', right_index=True)
df_trump['polarity'] = merged_df.groupby('id')['polarity'].sum()
print(df_trump.head())

print('Most negative tweets:')
print(df_trump.sort_values(by='polarity')[['text', 'polarity']][:20])

print('Most positive tweets:')
print(df_trump.sort_values(by='polarity', ascending=False)[['text', 'polarity']][:20])

# Plot KDE of tweet sentiment (nyt vs fox)
nyt_tweets = df_trump[df_trump['text'].str.contains('nyt')]['polarity']
fox_tweets = df_trump[df_trump['text'].str.contains('fox')]['polarity']
sns.distplot(nyt_tweets, label='nyt')
sns.distplot(fox_tweets, label='fox')
plt.legend()
plt.show()

# create a dataframe called tmp to store all words appear in the tweets
tmp = tidy_format['word'].value_counts()

# remove stopwords
stop_words = set(stopwords.words('english'))
tmp = tmp[~tmp.index.isin(stop_words)]

# deal with plurals
lemmatizer = WordNetLemmatizer()
tmp.index = tmp.index.map(lambda word: lemmatizer.lemmatize(word))

# Remove numbers
tmp = tmp[~tmp.index.str.isnumeric()]

# Remove words with only 1 or 2 length
tmp = tmp[tmp.index.str.len() > 2]

# Remove political words
remove_words = ['http', 'amp', 'trump', 'hillary', 'realdonaldtrump', 'clinton', 'trump2016', 'maga']
tmp = tmp[~tmp.index.isin(remove_words)]
top50_words = tmp.index[:50].to_list()
print(top50_words)

# Create document-frequency matrix
w_to_idx = {top50_words[i]: i for i in range(len(top50_words))}
X = np.zeros((5000, 50))
for i in range(5000):
    for word in df_trump.iloc[i]['no_punc'].split():
        if word in top50_words:
            X[i, w_to_idx[word]] += 1
print(X[:10, :])

# Find the first 50 PCA's for the document-frequency matrix
pca = PCA(n_components=50)
pca.fit(X)
components = pca.components_
print(components)

# Plot the heatmap of the first 50 PCA's
pcalabel = []
for i in range(1, 51):
    pcalabel.append('PC' + str(i))
plt.figure(figsize=(15,15))
cmap = sns.diverging_palette(100, 400, as_cmap=True)
sns.heatmap(components, cmap=cmap, yticklabels=top50_words, xticklabels=pcalabel, vmin=-1, vmax=1)
plt.show()

# Compare the first two PCA's
pc1 = components[0]
pc2 = components[1]
fig = sns.jointplot(x=pc2, y=pc1)
fig.set_axis_labels('PC2', 'PC1', fontsize=16)

# Find the top 20 words with the highest average retweet count
merged_df = tidy_format.merge(df_trump, how='left', left_index=True, right_index=True)
word_id_counts = merged_df.reset_index().groupby('word')['id'].nunique()
merged_df_filtered = merged_df[merged_df['word'].isin(word_id_counts[word_id_counts >= 25].index)]
retweets = merged_df_filtered.groupby('word')['retweet_count'].mean()
top_20_retweets = pd.DataFrame(retweets).sort_values(ascending=False, by='retweet_count').head(20)
print(top_20_retweets)

# Plot the top 20 words with the highest average retweet count
top_20_retweets.plot(kind='bar')
plt.title('Top 20 Words with Highest Retweet Count')
plt.ylabel('Retweet Count')
plt.xlabel('Word')
plt.show()