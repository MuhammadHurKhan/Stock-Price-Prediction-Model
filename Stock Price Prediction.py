#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Take input of user as ticker
ticker = input("Enter the ticker symbol of the stock you want to predict: ")

# Load the data using yfinance
data = yf.download(ticker, period="max")

# Sort the data by date
data = data.sort_values('Date')

# Create a new dataframe with only the 'Close' column
data = data.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = int(np.ceil(0.9 * len(dataset)))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data
train_data = scaled_data[0:training_data_len, :]

# Split the data into x_train and y_train datasets
x_train = []
y_train = []
time_steps = 60

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i - time_steps:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data
test_data = scaled_data[training_data_len - time_steps:, :]

# Create the x_test and y_test datasets
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i - time_steps:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'), row=1, col=1)
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual'), row=1, col=1)
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted'), row=1, col=1)

fig.update_layout(title=f"Stock Price Prediction for {ticker.upper()}", xaxis_title="Date", yaxis_title="Close Price")


# In[ ]:




