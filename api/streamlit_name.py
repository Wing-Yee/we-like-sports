import streamlit as st
import numpy as np #use to convert image
import json #for when we have real results
import requests #for when we have real results
from PIL import Image

test_image = Image.open('/Users/sun/we-like-sports/api/wordcloudtest.jpg')

st.title("We Like Sports Recommendation")
st.write("")
st.write("Select the importance of the below features from 1-5")
st.write("1 being least important, 5 being most important")

sentiment = st.slider("Sentiment", 1,5,1)
coverage = st.slider("Coverage", 1,5,1)
engagement = st.slider("Engagement", 1,5,1)

if st.button('Calculate'):
    #res = requests.post(url = http://localhost:8000/calculate, data = json.dumps(X_pred_conv) )
    st.subheader(f"Your recommended Sport event is...The EFL ") #{res.text} ")
    st.image(test_image, caption='Top used words for this event')
