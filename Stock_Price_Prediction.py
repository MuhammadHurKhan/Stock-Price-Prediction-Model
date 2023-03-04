#!/usr/bin/env python
# coding: utf-8

# In[31]:

import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define the function to retrieve stock data
def load_data(ticker):
    start_date = dt.datetime.now() - dt.timedelta(days=365)
    end_date = dt.datetime.now()
    stock_data = yf.download(ticker, start_date, end_date)
    return stock_data

# Define the function to prepare the data for the LSTM model
def prepare_data(data, num_days):
    x_data = []
    y_data = []
    for i in range(num_days, len(data)):
        x_data.append(data[i - num_days:i, 0])
        y_data.append(data[i, 0])
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data

# Define the function to predict the next 7 days of stock prices using the LSTM model
def predict_stock_price(model, data, num_days):
    next_dates = []
    next_prices = []
    last_date = data.index[-1].date()
    for i in range(1, num_days + 1):
        next_date = last_date + dt.timedelta(days=i)
        next_dates.append(next_date)
        next_data = data['Close'][i-1:].values.reshape(-1, 1)
        next_data = np.concatenate((next_data, np.zeros((num_days-i, 1))))
        next_data_scaled = scaler.transform(next_data)
        next_data_scaled = np.reshape(next_data_scaled, (1, num_days, 1))
        next_price = model.predict(next_data_scaled)
        next_price = scaler.inverse_transform(next_price)
        next_prices.append(next_price[0][0])
    return next_dates, next_prices

# Load the LSTM model
model = load_model('lstm_model.h5')

# Define the number of days to use for the LSTM model
num_days = 60

# Define the app layout
st.set_page_config(page_title='Stock Price Prediction', layout='wide')
st.title('Stock Price Prediction')
st.sidebar.header('User Input')

# Get the user input
ticker = st.sidebar.text_input("Enter the stock ticker symbol (e.g. AAPL)", value='AAPL')
stock_data = load_data(ticker)

# Prepare the data for the LSTM model
scaler = load(open(f"{ticker}_scaler.pkl", "rb"))
test_data = stock_data['Close'].values.reshape(-1, 1)
test_data = scaler.transform(test_data)

# Get the predicted stock prices for the next 7 days
x_test, y_test = prepare_data(test_data, num_days)
next_dates, predicted_prices = predict_stock_price(model, stock_data, num_days)
next_dates_formatted = [date.strftime('%Y-%m-%d') for date in next_dates]

# Display the predicted stock prices in a table and a line chart
st.subheader("Expected Stock Prices for the Next 7 Days")
st.table(pd.DataFrame({'Date': next_dates_formatted, 'Price': predicted_prices}))

st.subheader("Predicted Prices for the Next 7 Days")
df = pd.DataFrame({'Date': next_dates_formatted, 'Price': predicted_prices})
df.set_index('Date', inplace=True)
st.line_chart(df)









# In[ ]:




