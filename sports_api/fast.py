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

import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

url = http://localhost:8501/
app = FastAPI()

@app.get("/")
def root():
    return {'greeting': 'Hello'}

# @app.get('/input')

#https://medium.com/codex/streamlit-fastapi-%EF%B8%8F-the-ingredients-you-need-for-your-next-data-science-recipe-ffbeb5f76a92

# @app.get("/calculate")
def calculate(sentiment: int,
            coverage: int,
            likes: int,
            comments: int,
            retweets: int):

    #convert params into X_pred dictionary format
    X_pred_conv = pd.DataFrame(dict(
            # key=[str(pickup_datetime.strftime('%Y-%m-%d %H:%M:%S UTC'))],
            #What to put for key?????
            sentiment=[int(sentiment)],
            coverage=[int(coverage)],
            likes=[int(likes)],
            comments=[int(comments)],
            retweets=[int(retweets)],
            ))

    #select event from backend
    from "sentiment_analysis_test.main" import select_event
    result_name = select_event(comments, likes, retweets, sentiment, coverage)

    # taxifare.interface.main import pred
    #predict = float(pred(X_pred_conv))

    #convert prediction into dictionary from numpy array
    # predict_dict = dict(enumerate(predict.flatten(), 1))
    return {'fare_amount' : 'hold' }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
