Stock Price Prediction 

This project provides a detailed exploration of stock price prediction. It uses an LSTM (long short-term memory) recurrent neural network to analyze the past stock prices of different stocks and make future price predictions. 


Data

The dataset used for this project was downloaded from Yahoo Finance. It contains the stock prices of US stocks over a period of time. 


Model

The LSTM model was built using the Keras library in Python. The model consists of two LSTM layers, each followed by a dropout layer, and two dense layers. The model was trained using the model.fit() function and optimized using the adam optimizer. 


Evaluation

The model was evaluated by comparing its predictions with the actual stock prices. The model achieved a Root Mean Squared Error of 1.48, indicating that it was successful in predicting stock prices. 


Summary

This project was successful in demonstrating the potential of LSTM models in predicting stock prices. The model was able to accurately predict future stock prices, with a RMSE of 1.48.
