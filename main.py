import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math

START = "2017-05-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Predict Stock Trends ')

stocks = ('TTM', 'RELI', 'IBM', 'SBIN.NS','MSFT')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
title= selected_stock
num_prediction  = st.slider('Days of prediction:', 10, 30)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
df = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(df.tail())


close_data = df['Close'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)



model = Sequential()

model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 15
model.fit(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))


trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground truth'
)
layout = go.Layout(
    title= selected_stock +" Stocks",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
st.plotly_chart(fig)
close_data = close_data.reshape((-1))

num_prediction=math.floor(num_prediction/10)*10
def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
trace1 = go.Scatter(
    x = df['Date'].tolist(),
    y = close_data,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Prediction'
)
layout = go.Layout(
    title= selected_stock +" Stocks",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
st.plotly_chart(fig)
