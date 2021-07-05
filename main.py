import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import pandas_datareader.data as pdr
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import yfinance as yf


yf.pdr_override()

# Load data

stock_ticker = 'NVDA'

start = '01-01-2012'
end = '01-01-2021'

start = dt.datetime.strptime(start, '%d-%m-%Y')
end = dt.datetime.strptime(end, '%d-%m-%Y')

#start = dt.datetime(2012, 1,1)
#end = dt.datetime(2020, 1, 1)

data = pdr.get_data_yahoo(stock_ticker, data_source='yahoo', start=start, end=end)
#data = web.DataReader(stock_ticker, 'yahoo', start, end)

# pre-processing

min_max = MinMaxScaler(feature_range=(0,1))
scaled_data = min_max.fit_transform(data['Close'].values.reshape(-1,1))

prediction_range = 60

x_train = []
y_train = []

for i in range(prediction_range, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_range:i, 0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# the model (LSTM layers follower by Dropout layers until a final Dense layer that gives us the stock prediction)

model = Sequential()

model.add(LSTM(units=45, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=45, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=45))
model.add(Dense(units=1))  # prediction of next closing value
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Testing data

# loading test data

#test_start = dt.datetime(2021,2,1)
#test_end = dt.datetime.now()
test_start = '01-02-2021'
test_end = '05-07-2021'

test_start = dt.datetime.strptime(test_start, '%d-%m-%Y')
test_end = dt.datetime.strptime(test_end, '%d-%m-%Y')

test_data = pdr.get_data_yahoo(stock_ticker, data_source='yahoo', test_start=start, test_end=end)
#test_data = web.DataReader(stock_ticker, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_data = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_data[len(total_data) - len(test_data) - prediction_range:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = min_max.transform(model_inputs)

# make predictions on test data

x_test = []

for i in range(prediction_range, len(model_inputs)):
    x_test.append(model_inputs[i-prediction_range:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
# prices are in transformed scale so need to inverse transform them back
predicted_prices = min_max.inverse_transform(predicted_prices)

# plot test predictions

plt.plot(actual_prices, color="blue", label="Actual {} price".format(stock_ticker))
plt.plot(predicted_prices, color="red", label="Predicted {} price".format(stock_ticker))
plt.title("{} stock price".format(stock_ticker))
plt.xlabel('Time')
plt.ylabel("{} stock price".format(stock_ticker))
plt.legend()
plt.show()


# predict next day

real_data = [model_inputs[len(model_inputs) + 1 - prediction_range:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = min_max.inverse_transform(prediction)
print("Prediction for {0}: {1}".format(stock_ticker, prediction))

