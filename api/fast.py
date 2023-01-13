#each time you want to run from your local terminal

#pyenv virtualenv 3.8.12  welikesports-env
#pip install --upgrade pip
#pyenv local welikesports-env
#code .

#in new tab of terminal
# python -m pip install fastapi
# pip install uvicorn

#uvicorn sports_api.api.fast:app --reload

#import pandas as pd
#from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import json
import requests

app = FastAPI()

@app.get("/")
def root():
    return {'greeting': 'Hello'}

@app.get('/input')
def input():
    st.title("We Like Sports Recommendation")
    st.write("")
    st.write("Select the importance of the below features from 1-5")
    st.write("1 being least important, 5 being most important")

    sentiment = st.slider("Sentiment", 1,5,1)
    coverage = st.slider("Coverage", 1,5,1)
    engagement = st.slider("Engagement", 1,5,1)

#https://medium.com/codex/streamlit-fastapi-%EF%B8%8F-the-ingredients-you-need-for-your-next-data-science-recipe-ffbeb5f76a92

@app.get("/predict")
def predict(sentiment: int,
            coverage: int,
            engagement: int):

    sentiment = st.slider("Sentiment", 1,5,1)
    coverage = st.slider("Coverage", 1,5,1)
    engagement = st.slider("Engagement", 1,5,1)

    #convert params into X_pred dictionary format
    X_pred_conv = pd.DataFrame(dict(
            # key=[str(pickup_datetime.strftime('%Y-%m-%d %H:%M:%S UTC'))],
            #What to put for key?????
            sentiment=[int(sentiment)],
            coverage=[int(coverage)],
            engagement=[int(engagement)]
            ))

    # predict y using X_pred
    from taxifare.interface.main import pred
    predict = float(pred(X_pred_conv))

    # convert prediction into dictionary from numpy array
    predict_dict = dict(enumerate(predict.flatten(), 1))
    return {'fare_amount' : predict }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
