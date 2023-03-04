#!/usr/bin/env python
# coding: utf-8

# In[31]:
import streamlit as st
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Set the page title and favicon
st.set_page_config(page_title='Stock Price Prediction App', page_icon=':money_with_wings:')

# Define a function to get the stock data
def get_data(ticker_symbol):
    # Get the stock data
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="max")
    return data

# Define a function to scale the data
def scale_data(data):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

# Define a function to get the last 7 days predictions
def get_next_dates():
    next_dates = []
    today = dt.datetime.today()
    for i in range(7):
        next_dates.append(today + dt.timedelta(days=i))
    next_dates = pd.DataFrame(next_dates)
    next_dates.columns = ['Date']
    return next_dates

# Define a function to create the LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define a function to train the LSTM model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

# Define a function to make predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Get the stock ticker from the user
ticker_symbol = st.text_input('Enter Ticker Symbol', value='AAPL')

# Get the stock data
try:
    data = get_data(ticker_symbol)
except:
    st.error('Invalid Ticker Symbol, please try again.')
    st.stop()

# Scale the data
scaler, scaled_data = scale_data(data)

# Get the last 7 days predictions
next_dates = get_next_dates()
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prepare the data for training
train_data = scaled_data[:-60]
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Load the LSTM model
model = create_model(X_train)

# Train the LSTM model
train_model(model, X_train, y_train)

# Make predictions
predictions = make_predictions(model, X_test, scaler)


# Plot the predicted prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=next_dates['Date'], y=predictions.flatten(),
                    mode='lines+markers',
                    name='Predicted Prices'))
fig.update_layout(title=f"Predicted Prices for {ticker_symbol}",
                  xaxis_title="Date",
                  yaxis_title="Price")
st.plotly_chart(fig)









# In[ ]:




