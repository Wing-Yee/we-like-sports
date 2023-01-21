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
import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#results dataframe
df2 = pd.read_csv('sentiment_analysis_score.csv')
df2 = df2.drop(columns = 'Sentiment')

app = FastAPI()

#https://medium.com/codex/streamlit-fastapi-%EF%B8%8F-the-ingredients-you-need-for-your-next-data-science-recipe-ffbeb5f76a92

@app.get("calculate")
def calculate(sentiment: st.slider("Sentiment", 0,5,1),
            coverage: st.slider("Coverage", 0,5,1),
            likes: st.slider("Likes", 0,5,1),
            comments: st.slider("Comments", 0,5,1),
            retweets: st.slider("Retweets", 0,5,1)):


    sentiment = st.slider("Sentiment", 0,5,1)
    coverage = st.slider("Coverage", 0,5,1)
    likes = st.slider("Likes", 0,5,1)
    comments = st.slider("Comments", 0,5,1)
    retweets = st.slider("Retweets", 0,5,1)

#     #convert params dictionary format
#     user_input_df = pd.DataFrame(dict(
#             sentiment=[int(sentiment)],
#             coverage=[int(coverage)],
#             likes=[int(likes)],
#             comments=[int(comments)],
#             retweets=[int(retweets)]
#             ))

#     return {'test' : user_input_df }

#     output_name = select_event(user_input_df['comments'],
#             user_input_df['likes'],
#             user_input_df['retweets'],
#             user_input_df['sentiment'],
#             user_input_df['coverage'])

#     return output_name


# def select_event(comments_score, likes_score, retweet_score, compound_score, coverage_score):
#     df2['comments_weighted'] = df2['Comments'].apply(lambda x: x*comments_score)

#     df2['likes_weighted'] = df2['Likes'].apply(lambda x: x*likes_score)

#     df2['retweet_weighted'] = df2['Retweets'].apply(lambda x: x*retweet_score)

#     df2['compound_weighted'] = df2['Compound'].apply(lambda x: x*compound_score)

#     df2['coverage_weighted'] = df2['Tidy_tweets'].apply(lambda x: x*coverage_score)

#     df2['final_weighted'] = df2['comments_weighted'] + df2['likes_weighted'] + df2['retweet_weighted'] + df2['compound_weighted'] + df2['coverage_weighted']

#     return df2.iloc[df2.final_weighted.argmax(),0]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
