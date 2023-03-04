#!/usr/bin/env python
# coding: utf-8

# In[31]:


import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, LSTM
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Function to create a LSTM model
def create_model(train_data):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to preprocess the data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
    data.index = data['Date']
    data.drop(['Date'], axis=1, inplace=True)
    data = data.sort_index(ascending=True, axis=0)
    return data

# Function to create the training and testing data sets
def create_train_test_data(data, train_data_size):
    train_data = data[0:train_data_size, :]
    x_train = []
    y_train = []
    for i in range(60, train_data_size):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Function to make predictions
def make_predictions(model, x_test):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Get the stock data
ticker = st.text_input("Enter stock ticker symbol (e.g. AAPL for Apple): ")
stock_data = yf.download(ticker, period="max")

# Preprocess the data
data = preprocess_data(stock_data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the training and testing data sets
train_data_size = int(len(scaled_data) * 0.8)
x_train, y_train = create_train_test_data(scaled_data, train_data_size)

# Create the LSTM model
model = create_model(x_train)

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Make predictions
test_data = scaled_data[train_data_size - 60: , :]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = make_predictions(model, x_test)

# Show the prediction graph
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=data.index[train_data_size:], y=data['Close'][train_data_size:], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=data.index[train_data_size:], y=predictions[:,0], mode='lines', name='Predicted'))
fig.update_layout(title=f"Stock Price Prediction for {ticker.upper()}",
xaxis_title="Date",
yaxis_title="Stock Price ($)",
xaxis_rangeslider_visible=True)
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode="lines", name="Future Predictions"))
st.plotly_chart(fig)


# In[ ]:




