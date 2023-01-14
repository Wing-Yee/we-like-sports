import streamlit as st
import numpy as np #use to convert image
import json #for when we have real results
import requests #for when we have real results
from PIL import Image
import pandas as pd

#Load images and dataframes
word_cloud = Image.open('wordcloudtest.jpg')
event_icon = Image.open('EFL.png')
result_table = pd.read_csv('test_results_table.csv',  encoding= 'unicode_escape')
comparison_table = pd.read_csv('test_results_table.csv',  encoding= 'unicode_escape')

#UI for user input
st.title("We Like Sports Recommendation")
st.write("[![Star](https://img.shields.io/github/stars/Wing-Yee/we-like-sports.svg?logo=github&style=social)](https://gitHub.com/Wing-Yee/we-like-sports)")
st.write("")
st.subheader("Rate the below dimensions based on their importance from 1-5")
st.subheader("1 being least important, 5 being most important")

sentiment = st.slider("Sentiment", 1,5,1)
coverage = st.slider("Coverage", 1,5,1)
likes = st.slider("Likes", 1,5,1)
comments = st.slider("Comments", 1,5,1)
retweets = st.slider("Retweets", 1,5,1)

#UI for output
if st.button('Calculate'):
    #res = requests.post(url = http://localhost:8000/calculate, data = json.dumps(X_pred_conv) )
    st.subheader(f"Your recommended Sport event is...The EFL ") #{res.text} ")

    st.write(f"Event Metrics")
    st.dataframe(result_table)

    st.write(f"Comparisons Event Metrics")
    st.dataframe(comparison_table)

    col1, col2 = st.columns((2,1))
    with col1:
        st.image(word_cloud, caption='Top used words for this event')
    with col2:
        st.image(event_icon)
