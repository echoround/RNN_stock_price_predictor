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

