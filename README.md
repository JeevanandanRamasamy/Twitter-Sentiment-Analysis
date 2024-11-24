# Twitter Sentiment Analysis

This project provides an in-depth analysis of Donald Trump’s tweets using data collected from Twitter's Python API (Tweepy). It includes data cleaning, sentiment analysis, visualization, and dimensionality reduction techniques such as PCA. The analysis explores tweet sources, tweet frequency patterns, sentiment trends, and popular words from Trump’s tweets. Additionally, we use PCA to examine relationships between different words used in the tweets.

## Requirements

This project requires several Python libraries: 

- numpy
- pandas
- matplotlib
- seaborn
- nltk
- sklearn

To install the required dependencies, you can use the following command:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

## Required Data
- **Tweets Data**: The primary dataset consists of tweets from Donald Trump’s Twitter account. To collect the tweets, you can use the Tweepy library, which allows you to access the Twitter API and retrieve historical tweets. You will need to create a Twitter Developer account and obtain your API credentials (API key, API secret key, access token, and access token secret).
- **VADER Sentiment Lexicon**: You will also need the **VADER Sentiment Lexicon** (`vader_lexicon.txt`) to perform sentiment analysis on the tweets. This lexicon is available from the VADER Sentiment Analysis GitHub repository. Download the lexicon and save it as `vader_lexicon.txt` in your project directory to use for sentiment analysis.

---

## Data Cleaning

The dataset is cleaned and preprocessed as follows:
1. **Remove duplicates**: Duplicate tweets are removed using the tweet id.
2. **Convert date and time**: The created_at field is converted to a datetime object and sorted chronologically.
3. **Text cleaning**: HTML tags from the source field are removed, and stopwords and punctuation are filtered out from tweet text.
4. **Time zone conversion**: The timestamps are converted to Eastern Standard Time (EST).

## Sentiment Analysis

The sentiment of each tweet is calculated using a predefined VADER sentiment lexicon. Positive or negative sentiment scores are computed based on the words present in each tweet, with a sentiment score being summed over all words.

Example of Sentiment Calculation:
```python3
df_trump['sentiment'] = df_trump['text'].apply(get_sentiment)
```

## Data Visualizations

The following visualizations are generated:
1. **Source of Tweets**: A bar chart showing the number of tweets sent from different sources (e.g., iPhone, Android).
2. **Tweet Frequency by Hour**: A bar chart showing how frequently tweets are posted by hour.
3. **Sentiment Distribution**: KDE plots comparing the sentiment of tweets from specific sources or topics (e.g., tweets containing ‘nyt’ vs. ‘fox’).
4. **Word Frequency**: A bar chart displaying the top 50 most frequent words in the tweets, excluding stopwords and common political words.
5. **PCA Analysis**: A heatmap visualizing the first 50 Principal Components (PCs) of a document-frequency matrix built from the top 50 words, showing the relationships between words in the tweet corpus.
6. **Top Words by Retweet Count**: A bar chart of the top 20 most retweeted words in Trump’s tweets.


