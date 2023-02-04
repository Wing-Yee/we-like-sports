##once youve loaded fastpy open another tab
# streamlit run sports_app/streamlit_name.py

import streamlit as st
import numpy as np #use to convert image
import json #for when we have real results
import requests #for when we have real results
from PIL import Image
import pandas as pd
import sys, os

#path to csv file
abs_csv_path = os.path.join(os.path.dirname(__file__), '../sports_api/sentiment_analysis_score.csv')
result_table = pd.read_csv(abs_csv_path,  encoding= 'unicode_escape')


# #UI for user input
st.title("We Like Sports Recommendation")
st.write("[![Star](https://img.shields.io/github/stars/Wing-Yee/we-like-sports.svg?logo=github&style=social)](https://gitHub.com/Wing-Yee/we-like-sports)")
st.write("")
st.subheader("Rate the below dimensions based on their importance from 0-5")
st.subheader("0 being least important, 5 being most important")

#slide inputs
sentiment = st.slider("Sentiment", 0,5,1)
coverage = st.slider("Coverage", 0,5,1)
likes = st.slider("Likes", 0,5,1)
comments = st.slider("Comments", 0,5,1)
retweets = st.slider("Retweets", 0,5,1)

#?sentiment=1&coverage=1&likes=1&comments=1&retweets=1

# #UI for output
if st.button('Calculate'):
    #make dictionary
    params = {
    'sentiment':sentiment,
    'coverage':coverage,
    'likes':likes,
    'comments':comments,
    'retweets':retweets,
    }

    url = 'http://127.0.0.1:8000/calculate'
    response = requests.get(url, params=params)
    return_result = response.json()
    result_name = return_result['event name']

    ##Locate images and csvs from folder and open
    event_name = os.path.join(os.path.dirname(__file__), f'../sports_api/{result_name}.png')
    word_cloud = Image.open(event_name)

    logo_name = os.path.join(os.path.dirname(__file__), f'../sports_api/{result_name} logo.png')
    event_icon = Image.open(logo_name)

    st.subheader(f"Your recommended Sport event is... {result_name} ") #{res.text} ")

    st.write(f"Event Metrics")
    st.dataframe(result_table.loc[result_table['source'] == result_name])

    st.write(f"Comparisons Event Metrics")
    st.dataframe(result_table.loc[result_table['source'] != result_name])

    col1, col2 = st.columns((2,1))
    with col1:
       st.image(word_cloud, caption='Top used words for this event')
    with col2:
       st.image(event_icon)
