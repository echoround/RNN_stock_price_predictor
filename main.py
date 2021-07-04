import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data

stock_ticker = "NVDA"

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(stock_ticker, 'yahoo', start, end)

# pre-processing

min_max = MinMaxScaler(feature_range=(0,1))
scaled_data = min_max.fit_transform(data['Close'].values.reshape(-1,1))

prediction_range = 60

x_train = []
y_train = []

for i in range(prediction_range, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_range:i, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# the model (LSTM layers follower by Droput layers until a final Dense layer that gives us the stock prediction)

model = Sequential()

model.add(LSTM(units=45, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=45, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=45))
model.add(Dense(units=1)) #prediction of next closing value






