#!/usr/bin/env python
# coding: utf-8

# In[31]:

import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def search_stock(query):
    stock = yf.Ticker(query)
    return stock.info['symbol']

def get_stock_data(ticker_symbol, days):
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    stock_data = yf.download(ticker_symbol, start=start_date, end=today)
    return stock_data

def predict_stock_price(stock_data, days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    prediction_days = 7
    x_test = []
    for x in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[x-prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_test.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    
    last_date = stock_data.iloc[-1].name
    next_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    next_dates_formatted = [date.strftime('%Y-%m-%d') for date in next_dates]
    
    predicted_prices = []
    for i in range(7):
        x_input = scaled_data[-prediction_days:]
        x_input = np.reshape(x_input, (1, prediction_days, 1))
        y_pred = model.predict(x_input)
        y_pred = scaler.inverse_transform(y_pred)
        predicted_prices.append(y_pred[0][0])
        scaled_data = np.vstack((scaled_data, y_pred))
    
    return next_dates_formatted, predicted_prices

st.title("Stock Price Prediction App")

search_query = st.text_input("Search for a stock by name:")
if search_query:
    ticker_symbol = search_stock(search_query)
    if not ticker_symbol:
        st.error("Sorry, no stock found for the given search query.")
    else:
        st.success(f"Ticker symbol for {search_query}: {ticker_symbol}")
        num_days = st.slider("Select the number of days of historical data to use:", min_value=30, max_value=365, value=90)
        stock_data = get_stock_data(ticker_symbol, num_days)
        next_dates, predicted_prices = predict_stock_price(stock_data, num_days)

        st.subheader("Expected Stock Prices for the Next 7 Days")
        st.table(pd.DataFrame({'Date': next_dates, 'Price': predicted_prices}))

       # Add line chart for predicted prices
        st.subheader("Predicted Prices for the Next 7 Days")
        df = pd.DataFrame({'Date': next_dates_formatted, 'Price': predicted_prices})
        df.set_index('Date', inplace=True)
        st.line_chart(df)










# In[ ]:




