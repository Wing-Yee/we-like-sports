##the first time you run you need to create the virtual environment
# cd into the we-like-sports folder in your terminal
# pyenv virtualenv 3.8.12  .env
## then in VS code, command + p and type 'python select interpretor'
## choose the env you just made!

##each time you want to run from your local terminal

#pyenv virtualenv 3.8.12  we-like-sports-env
#pip install --upgrade pip
#pip install streamlit
#pyenv local we-like-sports-env
#code .

##in new tab of terminal
# python -m pip install fastapi
# pip install uvicorn

#uvicorn sports_api.fast:app --reload

import pandas as pd
import csv
import os
import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sports_api.sentiment_analysis_test import select_event

#load dataframe with results
#path to csv file
abs_csv_path = os.path.join(os.path.dirname(__file__), '../sports_api/sentiment_analysis_score.csv')
df2 = pd.read_csv(abs_csv_path,  encoding= 'unicode_escape')

print(df2)

app = FastAPI()

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END


#@app.post("/check")
#def foo(file: UploadFile):
#    df = pd.read_csv(file.file)
#    return len(df)

@app.get("/calculate")
def calculate(sentiment: int,
            coverage: int,
            likes: int,
            comments: int,
            retweets: int):

    #select event from backend
    result_name = select_event(comments, likes, retweets, sentiment, coverage, df2)

    #convert prediction into dictionary from numpy array
    # predict_dict = dict(enumerate(predict.flatten(), 1))
    return {'event name' : str(result_name) }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
