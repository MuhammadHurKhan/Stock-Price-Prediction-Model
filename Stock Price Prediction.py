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
import streamlit as st
import concurrent.futures

def main():
    # Set the title of the app
    st.title("Stock Price Prediction")

    # Take input of user as ticker
    ticker = st.text_input("Enter the ticker symbol of the stock you want to predict: ")

    if not ticker:
        st.warning("Please enter a ticker symbol.")
        return

    # Load the data using yfinance
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(yf.download, ticker, period="6M")
        data = future.result()

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
    time_steps = 120

    for i in range(time_steps, len(train_data)):
        x_train.append(train_data[i - time_steps:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dense(20))
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
    x_test = np.reshape(x_test, (x_test.shape[1], x_test.shape[0], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Extend the data to include the next 
    num_days = 7
    last_date = data.index[-1]
    new_dates = pd.date_range(last_date, periods=num_days, freq='B')
    new_data = pd.DataFrame(index=new_dates, columns=data.columns)
    # Use the last available observation to fill missing values in the extended data
    new_data.fillna(method='ffill', inplace=True)


    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Create the x_test dataset for the new data
    x_test_new = []
    for i in range(time_steps, new_data_scaled.shape[0]):
        x_test_new.append(new_data_scaled[i - time_steps:i, 0])

    # Convert the data to a numpy array
    x_test_new = np.array(x_test_new)

    # Reshape the data
    x_test_new = np.reshape(x_test_new, (x_test_new.shape[1], x_test_new.shape[0], 1))

    # Get the model's predicted price values for the new data
    predictions_new = model.predict(x_test_new)
    predictions_new = scaler.inverse_transform(predictions_new)

    # Combine the original and new data
    combined_data = pd.concat([data, new_data])

    # Add the predicted values to the new data
    predictions_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=num_days, freq='B')
    predictions_df = pd.DataFrame(data=predictions_new, index=predictions_index, columns=data.columns)
    combined_data = pd.concat([combined_data, predictions_df])

    # Compute the moving average for the combined data
    window_size = 7
    combined_data['MA'] = combined_data['Close'].rolling(window=window_size).mean()

    # Plot the original and extended data with moving average
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close'], name='Extended'), row=1, col=1)
    fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['MA'], name=f'{window_size}-day MA'), row=2, col=1)
    fig.update_layout(xaxis_title='Date', title='Stock Prices')
    st.plotly_chart(fig)
main()

# In[ ]:




