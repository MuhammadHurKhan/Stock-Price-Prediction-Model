#!/usr/bin/env python
# coding: utf-8

# In[31]:

import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

# Set page title and favicon
st.set_page_config(page_title='Stock Price Prediction', page_icon=':money_with_wings:')

# Define function to preprocess the data
def preprocess_data(stock_data):
    data = pd.DataFrame(stock_data)
    data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
    data.index = data['Date']
    data.drop('Date', axis=1, inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    x_train, y_train = [], []
    for i in range(60,len(data)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

# Define function to create the LSTM model
def create_model(x_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Define function to make the prediction
def make_prediction(model, x_train, scaler, days):
    prediction_data = x_train[-1]
    predicted_prices = []
    for _ in range(days):
        prediction = model.predict(prediction_data.reshape(1,60,1))
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        predicted_prices.append(predicted_price)
        prediction_data = np.append(prediction_data[1:], prediction[0])
    return predicted_prices

# Set up the app layout
st.title("Stock Price Prediction")
st.sidebar.header("Enter the Stock Ticker")
ticker = st.sidebar.text_input("Ticker", "AAPL")

# Fetch the data
stock_data = yf.download(ticker, start="2010-01-01", end="2022-01-01")

# Preprocess the data
x_train, y_train, scaler = preprocess_data(stock_data)

# Create the LSTM model
model = create_model(x_train)

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Make the prediction
days = 7
predicted_prices = make_prediction(model, x_train, scaler, days)

# Create the plot
data = stock_data.filter(['Close'])
last_date = data.index[-1]
date_range = pd.date_range(last_date, periods=days+1, freq='B')
date_range = date_range[1:]
predicted_data = pd.DataFrame(data=predicted_prices, index=date_range, columns=['Prediction'])
data = pd.concat([data, predicted_data], axis=0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['Prediction'], name='Predicted Price'))

#Update the plot layout
fig.update_layout(title=f"Stock Price Prediction for {ticker.upper()}",
xaxis_title="Date",
yaxis_title="Stock Price ($)")

#Plot the predicted values
fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(),
mode="lines",
name="Predicted Price"))

#Show the plot
st.plotly_chart(fig)









# In[ ]:




