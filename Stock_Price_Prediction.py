#!/usr/bin/env python
# coding: utf-8

# In[31]:

import math
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import yfinance as yf

#Define function to create the model
def create_model(train_data, look_back):
  X, y = [], []
for i in range(len(train_data)-look_back-1):
  a = train_data[i:(i+look_back), 0]
  X.append(a)
  y.append(train_data[i + look_back, 0])
  X_train, y_train = np.array(X), np.array(y)

# Reshape input data to be 3D [samples, timesteps, features]
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(units=4, input_shape=(None, 1)))
  model.add(Dense(units=1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

return model

#Get the stock symbol from user
symbol = st.text_input('Enter stock symbol (e.g. AAPL for Apple Inc.):')

#Load the dataset
data = yf.download(symbol, period="10y")
df = pd.DataFrame(data=data, columns=['Close'])

#Create a scaler to normalize the data
scaler = MinMaxScaler(feature_range=(0,1))

#Reshape the data to 1D
close_price = df['Close'].values.reshape(-1,1)

#Normalize the data
scaled_close_price = scaler.fit_transform(close_price)

#Define the train and test data
train_size = int(len(scaled_close_price) * 0.67)
test_size = len(scaled_close_price) - train_size
train_data = scaled_close_price[0:train_size,:]
test_data = scaled_close_price[train_size:len(scaled_close_price),:]

#Define the number of previous days to use to predict the next day's closing price
look_back = 30

#Create and fit the LSTM model
model = create_model(train_data, look_back)

#Test the model
X_test, y_test = [], []
for i in range(len(test_data)-look_back-1):
a = test_data[i:(i+look_back), 0]
X_test.append(a)
y_test.append(test_data[i + look_back, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

#Reshape input data to be 3D [samples, timesteps, features]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Make predictions on test data
predictions = model.predict(X_test)

#Invert the predictions and actual values to their original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

#Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)

#Calculate the root mean squared error
rmse = math.sqrt(mse)

#Plot the predicted vs. actual closing prices
st.line_chart(pd.DataFrame({'Actual': y_test.reshape(-1), 'Predicted': predictions.reshape(-1)}))

#Print the root mean squared error
st.write('Root Mean Squared Error:', rmse)








# In[ ]:




