#!/usr/bin/env python
# coding: utf-8
# ## Text Pre-processing
import numpy as np
import pandas as pd
import csv

df2 = pd.read_csv('sports_api/sentiment_analysis_score.csv')

def select_event(comments_score, likes_score, retweet_score, compound_score, coverage_score, df2):

    df2['comments_weighted'] = df2['Comments'].apply(lambda x: x*comments_score)

    df2['likes_weighted'] = df2['Likes'].apply(lambda x: x*likes_score)

    df2['retweet_weighted'] = df2['Retweets'].apply(lambda x: x*retweet_score)

    df2['compound_weighted'] = df2['Compound'].apply(lambda x: x*compound_score)

    df2['coverage_weighted'] = df2['Tidy_tweets'].apply(lambda x: x*coverage_score)

    df2['final_weighted'] = df2['comments_weighted'] + df2['likes_weighted'] + df2['retweet_weighted'] + df2['compound_weighted'] + df2['coverage_weighted']

    result_name = df2.iloc[df2.final_weighted.argmax(),0]

    return result_name


if __name__ == '__main__' :
    print(select_event(1,2,3,4,5,df2))
