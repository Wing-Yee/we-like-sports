#!/usr/bin/env python
# coding: utf-8

# ## Text Pre-processing
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import time
import string
import warnings
import glob
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# for all NLP related operations on text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# To mock web-browser and scrap tweets
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# To consume Twitter's API
import tweepy
from tweepy import OAuthHandler 

# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

# ignoring all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# downloading stopwords corpus
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')
nltk.download('stopwords')
stopwords = set(stopwords.words("english"))

# for showing all the plots inline
%matplotlib inline


# ## Merge all .csv files

def merge_files():

    path = 'raw data'
    file_list = os.listdir('raw data')

    full_path = []

    for f in file_list:
        full_path.append(os.path.join(path,f))

    excl_list = []
    for file in full_path:
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            df['source'] = file
            excl_list.append(df)

    # concatenate all DataFrames in the list
    # into a single DataFrame, returns new DataFrame.
    excl_merged = pd.concat(excl_list, ignore_index=True)
 
    # exports the dataframe into excel file
    # with specified name.
    excl_merged.to_csv('combine_test.csv', index=False)

    return

def get_data():

    tweets_df = pd.read_csv('combine_test.csv')

    return tweets_df

    # Get Event Name

def data_cleaning():

    tweets_df = get_data()

    #get event name
    tweets_df['source'] = tweets_df['source'].replace(regex=r"\_.*",value="").str.replace('raw data/','')
    
    tweets_df['tidy_tweets'] = tweets_df['Embedded_text'].str.replace("[^a-zA-Z# ]", "")
    
    #creat tidy_tweets for cleaned data
    tweets_df = tweets_df[tweets_df['tidy_tweets']!='']
    
    #remove nan
    tweets_df = tweets_df.astype(object).replace(np.nan, '0')

    return tweets_df



def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    return text 

#sentiment analysis

def sentiment_analysis():

    tweets_df = data_cleaning()

    tweets_df['tidy_tweets'] = np.vectorize(remove_pattern)(tweets_df['tidy_tweets'], "@[\w]*: | *RT*")

    cleaned_tweets = []

    for index, row in tweets_df.iterrows():
        # Here we are filtering out all the words that contains link
        words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
        cleaned_tweets.append(' '.join(words_without_links))

    tweets_df['tidy_tweets'] = cleaned_tweets

    tweets_df = tweets_df.drop_duplicates(subset=['tidy_tweets'], keep=False)
    
    tweets_df = tweets_df.reset_index(drop=True)

    stopwords_set = set(stopwords)

    cleaned_tweets = []

    for index, row in tweets_df.iterrows():
        # Here we are filtering out all the words that contains link
        words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
        cleaned_tweets.append(' '.join(words_without_links))

    tweets_df['tidy_tweets'] = cleaned_tweets

    tokenized_tweet = tweets_df['tidy_tweets'].apply(lambda x: x.split())


    word_lemmatizer = WordNetLemmatizer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

    for i, tokens in enumerate(tokenized_tweet):
        tokenized_tweet[i] = ' '.join(tokens)

    tweets_df['tidy_tweets'] = tokenized_tweet

    sentiments_using_SIA = tweets_df.tidy_tweets.apply(lambda tweet: fetch_sentiment_using_SIA(tweet))

    tweets_df['sentiment'] = sentiments_using_SIA

    tweets_df['sentiment_score'] = tweets_df.tidy_tweets.apply(lambda tweet: fetch_sentiment_score(tweet))

    tweets_df["sentiment_score"].apply(pd.Series)


    tweets_df = pd.concat([tweets_df, tweets_df["sentiment_score"].apply(pd.Series)], axis=1)

    tweets_df = tweets_df.drop(columns="sentiment_score")
    
    tweets_df['sen_pos'] = 1
    tweets_df.loc[tweets_df['sentiment'] == 'neg', "sen_pos"] = 0

    return tweets_df

def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return 'neg' if polarity_scores['neg'] > polarity_scores['pos'] else 'pos'

def fetch_sentiment_score(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return polarity_scores

def combined_table():

    tweets_df = sentiment_analysis()

    tweets_df['Timestamp'] = tweets_df['Timestamp'].astype('datetime64[ns]')
    tweets_df['Likes'] = tweets_df['Likes'].str.replace(',','').str.replace('K','').astype('float')
    tweets_df['Comments'] = tweets_df['Comments'].str.replace(',','').str.replace('K','').astype('float')
    tweets_df['Retweets'] = tweets_df['Retweets'].str.replace(',','').str.replace('K','').astype('float')

    tweets_df.to_csv('sentiment_analysis_test.csv')

    return tweets_df

# ## Get Sentiment Score
# reference for the defination of SIA score: https://analyticsindiamag.com/sentiment-analysis-made-easy-using-vader/

def score_table():

    tweets_df = combined_table()

    dfp = tweets_df.pivot_table(index = ['source'], 
                        values = ['tidy_tweets','Likes','Retweets','Comments','compound','neg','neu','pos','sen_pos'],
                        aggfunc = {'tidy_tweets':'count',
                                    'Likes':sum,
                                    'Retweets':sum,
                                    'Comments':sum,
                                    'compound':np.mean,
                                    'neg':np.mean,
                                    'neu':np.mean,
                                    'pos':np.mean,
                            "sen_pos":sum})
    dfp['compound'] = dfp['compound'].round(2)
    dfp['neu'] = dfp['neu'].round(2)
    dfp['pos'] = dfp['pos'].round(2)
    dfp['neg'] = dfp['neg'].round(2)

    dfp['sen_pos'] = (dfp['sen_pos']/dfp['tidy_tweets']).round(2)
    dfp['Comments'] = (dfp['Comments']/dfp['tidy_tweets']).round(2)
    dfp['LikeS'] = (dfp['Likes']/dfp['tidy_tweets']).round(2)
    dfp['Retweets'] = (dfp['Retweets']/dfp['tidy_tweets']).round(2)

    dfp.columns = [x.capitalize() for x in dfp.columns]

    return dfp

def scaled_table():

    scaler = MinMaxScaler()

    dfp = score_table()

    dfp[['Comments','Likes','Retweets','Tidy_tweets']] = scaler.fit_transform(dfp[['Comments','Likes','Retweets','Tidy_tweets']])

    return dfp

def select_event(comments_score, likes_score, retweet_score, compound_score, coverage_score):

    df2 = scaled_table()
    df2 = df2.drop(columns = 'Sentiment')

    df2['comments_weighted'] = df2['Comments'].apply(lambda x: x*comments_score)
    
    df2['likes_weighted'] = df2['Likes'].apply(lambda x: x*likes_score)

    df2['retweet_weighted'] = df2['Retweets'].apply(lambda x: x*retweet_score)

    df2['compound_weighted'] = df2['Compound'].apply(lambda x: x*compound_score)

    df2['coverage_weighted'] = df2['Tidy_tweets'].apply(lambda x: x*coverage_score)

    df2['final_weighted'] = df2['comments_weighted'] + df2['likes_weighted'] + df2['retweet_weighted'] + df2['compound_weighted'] + df2['coverage_weighted']
    
    return df2.iloc[df2.final_weighted.argmax(),0]




