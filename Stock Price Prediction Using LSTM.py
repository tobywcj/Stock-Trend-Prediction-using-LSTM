import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras


!pip install yfinance --upgrade --no-cache-dir -q

from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta

yf.pdr_override() # download data faster

start_date = '2010-01-01'
today_date = datetime.today().strftime('%Y-%m-%d')
print('Today\'s date: ', today_date)
tomorrow_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

# download dataframe
ticker = "AAPL"
raw_price_df = pdr.get_data_yahoo(ticker, start = start_date, end = tomorrow_date)

# Sort the data based on Date
price_df_sorted = raw_price_df.sort_values(by = ['Date'])
price_df_sorted

price_df_sorted.shape

# Check if Null values exist in stock prices data
price_df_sorted.isnull().sum()

# Get stock prices dataframe info
price_df_sorted.info()

price_df_sorted.describe()

# price_df_sorted.filter(['Close'])

price_df = price_df_sorted.reset_index()[['Date', 'Close']]
price_df

# Function to normalize stock prices based on their initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

# Function to plot interactive plots using Plotly Express
def interactive_plot(fig, df):
  for i in df.columns:
    if i != 'Date':
      fig.add_scatter(x = df['Date'], y = df[i], name = i)

# plot interactive chart for stocks data
import plotly.graph_objs as go

# create initial figure object
original_price_fig = go.Figure()
original_price_fig.update_layout(title = f'{ticker}Stock Price History', xaxis_title = 'Date', yaxis_title = 'Close Price USD ($)')

# plot initial data
interactive_plot(original_price_fig, price_df)
original_price_fig.show()


# # filter dataframe based on date
# filtered_df = price_df.loc[price_df['Date'] >= '2021-05-28']
# filtered_df

price_df

price_array = np.array(price_df['Close']).reshape(-1,1)
price_array.shape

# splitting dataset into train and test split this way, since order is important in time-series
# dont use train test split with it's default settings since it shuffles the data
def split_X_train_test(X, test_size=0.2):
  split = int((1-test_size) * len(X)) # round up the number
  X_train = X[:split]
  X_test = X[split:]

  return X_train, X_test

X_train_unscaled, X_test_unscaled = split_X_train_test(price_array, 0.01)

time_steps = 60

X_train_len = X_train_unscaled.shape[0]

X_train_unscaled.shape, X_test_unscaled.shape

X_train_unscaled[-1], X_test_unscaled[0]

X_test_unscaled = np.concatenate((X_train_unscaled[-time_steps:], X_test_unscaled), axis=0)

X_test_unscaled.shape, X_test_unscaled[time_steps-1:time_steps+1]

# LSTM are sensitive to the scale of the data. so we apply MinMax scaler (output = np.array format)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train_scaled = sc.fit_transform(X_train_unscaled)
X_test_scaled = sc.transform(X_test_unscaled)

X_train_scaled, X_train_scaled.shape

X_test_scaled, X_test_scaled.shape

import numpy as np

# convert an array of values into a dataset matrix
def create_dataset(array_data, time_steps=1):

  X, y = [], [] # become a list

  for i in range(time_steps, len(array_data)):
      X.append(array_data[i-time_steps:i, 0]) # its a 2D array => [row, col]
      y.append(array_data[i, 0])

  # return the new dataset
  return np.array(X), np.array(y) # convert into np array for RNN inputs

X_train, y_train = create_dataset(X_train_scaled, time_steps)
X_test, y_test = create_dataset(X_test_scaled, time_steps)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

  # RNN input params = (batch size, timesteps, features/indicators)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # add a new dimensionality to be compatible with the 3D tensor input shape of RNN and allow more indicators
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train.shape, y_train.shape

X_test.shape, y_test.shape

from google.colab import drive
drive.mount('/content/drive')

# from tensorflow.keras.models import load_model

# model = load_model(f'/content/drive/MyDrive/Projects/Stock Trend Prediction/models/{ticker}_model.h5')


from keras.models import Sequential # allow us to build a NN object representing a sequence of layers
from keras.layers import Dense # add the output layer
from keras.layers import LSTM # add the LSTM layers
from keras.layers import Dropout # add dropout regularization

# Create the model
model = Sequential() # regression -- predicting continuous values
model.add(LSTM(units = 150, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units = 150, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 150))
model.add(Dense(units = 1)) # dense class to create the fully-connected layer to fully connect the previous LSTM layer to one output unit

model.compile(optimizer='adam', loss="mean_squared_error")
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
file_path = f'/content/drive/MyDrive/Projects/Stock Trend Prediction/{ticker}_model.h5'
mc = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True)

# fit model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64, callbacks=[es, mc])

print(history.history.keys())

# Plot the training loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the validation loss
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

train_accuracy = model.evaluate(X_train, y_train, verbose=1)
print('train accuracy: ', train_accuracy)
test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print('test accuracy: ', test_accuracy)

# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import itertools


# def LSTM_HyperParameter_Tuning(config, x_train, y_train, x_test, y_test):

#     first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = config
#     possible_combinations = list(itertools.product(first_additional_layer, second_additional_layer, third_additional_layer,
#                                                   n_neurons, n_batch_size, dropout))

#     print(possible_combinations)
#     print('\n')

#     hist = []

#     for i in range(0, len(possible_combinations)):

#         print(f'{i+1}th combination: \n')
#         print('--------------------------------------------------------------------')

#         first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = possible_combinations[i]

#         # instantiating the model in the strategy scope creates the model on the TPU
#         #with tpu_strategy.scope():
#         regressor = Sequential()
#         regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
#         regressor.add(Dropout(dropout))

#         if first_additional_layer:
#             regressor.add(LSTM(units=n_neurons, return_sequences=True))
#             regressor.add(Dropout(dropout))

#         if second_additional_layer:
#             regressor.add(LSTM(units=n_neurons, return_sequences=True))
#             regressor.add(Dropout(dropout))

#         if third_additional_layer:
#             regressor.add(LSTM(units=n_neurons, return_sequences=True))
#             regressor.add(Dropout(dropout))

#         regressor.add(LSTM(units=n_neurons, return_sequences=False))
#         regressor.add(Dropout(dropout))
#         regressor.add(Dense(units=1, activation='linear'))
#         regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

#         es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#         '''''
#         From the mentioned article above --> If a validation dataset is specified to the fit() function via the validation_data or v
#         alidation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
#         '''''

#         file_path = '/content/drive/MyDrive/Projects/Stock Trend Prediction/best_model.h5'

#         mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

#         '''''
#         cb = Callback(...)  # First, callbacks must be instantiated.
#         cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
#         model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
#         '''''

#         regressor.fit(x_train, y_train, validation_split=0.2, epochs=40, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)

#         # load the best model
#         # regressor = load_model('best_model.h5')

#         train_accuracy = regressor.evaluate(x_train, y_train, verbose=0)
#         test_accuracy = regressor.evaluate(x_test, y_test, verbose=0)

#         hist.append(list((first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout,
#                           train_accuracy, test_accuracy)))

#         print(f'{str(i)}-th combination = {possible_combinations[i]} \n train accuracy: {train_accuracy} and test accuracy: {test_accuracy}')

#         print('--------------------------------------------------------------------')
#         print('--------------------------------------------------------------------')
#         print('--------------------------------------------------------------------')
#         print('--------------------------------------------------------------------')

#     return hist

# config = [[False, True], [False, True], [False, True], [16, 32], [8, 16, 32], [0.2]]

# # list of lists --> [[first_additional_layer], [second_additional_layer], [third_additional_layer], [n_neurons], [n_batch_size], [dropout]]

# hist = LSTM_HyperParameter_Tuning(config, X_train, y_train, X_test, y_test)  # change x_train shape

# hist = pd.DataFrame(hist)
# hist = hist.sort_values(by=[7], ascending=True)
# hist

# Make prediction
scaled_predictions = model.predict(X_test)

predictions_array = sc.inverse_transform(scaled_predictions)
predictions_array.shape

y_test = sc.inverse_transform(y_test.reshape(-1,1))

predictions_array.reshape(-1).shape, y_test.reshape(-1).shape

# Get the RMSE
rmse = np.sqrt(((predictions_array - y_test) ** 2).mean())
rmse

# Append the predicted values to the list
predictions_list = []

for i in predictions_array:
  predictions_list.append(i[0])

len(predictions_list)

train_df = price_df[:X_train_len]
train_df = train_df.rename(columns={'Close': 'Previous Prices'})

valid_df = price_df[X_train_len:]
valid_df = valid_df.rename(columns={'Close': 'Actual Closing Price'})
valid_df['Next Day Prediction'] = predictions_list

valid_df

train_df

# Plot the data
prediction_fig = go.Figure()
prediction_fig.update_layout(title = f"{ticker} Stock Trend Forecast", xaxis_title = 'Date', yaxis_title = 'Close Price USD ($)')

interactive_plot(prediction_fig, train_df)
interactive_plot(prediction_fig, valid_df)
prediction_fig.show()

from datetime import datetime, timedelta

def get_previous_and_next_open_days(date_str):
    # Convert the input date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d').date()

    # Define a list of public holidays
    public_holidays = [
        datetime(2023, 1, 2),   # New Year's Day (observed)
        datetime(2023, 1, 16),  # Martin Luther King Jr. Day
        datetime(2023, 2, 20),  # Presidents' Day
        datetime(2023, 4, 14),  # Good Friday
        datetime(2023, 5, 29),  # Memorial Day
        datetime(2023, 7, 4),   # Independence Day (observed)
        datetime(2023, 9, 4),   # Labor Day
        datetime(2023, 11, 23), # Thanksgiving Day
        datetime(2023, 12, 25)  # Christmas Day
    ]

    previous_open_day = date
    next_open_day = date

    # Calculate the previous and next available stock market open days
    if date.weekday() in [5, 6] or date in public_holidays:
        while previous_open_day.weekday() in [5, 6] or previous_open_day in public_holidays:
            previous_open_day -= timedelta(days=1)

        while next_open_day.weekday() in [5, 6] or next_open_day in public_holidays:
            next_open_day += timedelta(days=1)

    elif date.weekday() == 4:
        previous_open_day -= timedelta(days=1)
        next_open_day += timedelta(days=3)

    elif date.weekday() == 0:
        previous_open_day -= timedelta(days=3)
        next_open_day += timedelta(days=1)

    else:
        previous_open_day -= timedelta(days=1)
        next_open_day += timedelta(days=1)

    # Return the previous and next available stock market open days as strings
    return previous_open_day.strftime('%Y-%m-%d'), next_open_day.strftime('%Y-%m-%d')

import math

# Get the latest data, predicting yesterday's price
start_date = (datetime.today() - timedelta(days=(time_steps + math.ceil(time_steps/30 * 20 )))).strftime('%Y-%m-%d')
today_date = datetime.today().today().strftime('%Y-%m-%d')
prev_date, next_date = get_previous_and_next_open_days(today_date)
latest_df = pdr.get_data_yahoo(ticker, start = start_date, end = prev_date)
print('Dataset\'s Start date: ', start_date)
print('Today: ', today_date)
print('Yesterday: ', prev_date)
print('Tomorrow: ', next_date)
# print(latest_df.shape)

# Create a new dataframe
time_steps_df_sorted = latest_df.sort_values(by = ['Date'])
time_steps_df = time_steps_df_sorted.reset_index()[['Close']]
# Get the last 150 days closing price values and convert the dataframe to an array
time_steps_array = np.array(time_steps_df['Close'][-time_steps:]).reshape(-1,1)
# print(time_steps_array.shape)
# Normalization
X_test_quote = sc.transform(time_steps_array)
# Reshape the data
X_test_quote = np.reshape(X_test_quote, (1, X_test_quote.shape[0], X_test_quote.shape[1]))
# Get the predicted price
pred_quote = model.predict(X_test_quote)
# Unscaled
pred_quote = sc.inverse_transform(pred_quote)

print(f'{prev_date} predicted price: ', pred_quote[0][0])

quote = pdr.get_data_yahoo(ticker, start = prev_date, end = today_date)
quote['Close']

import pandas as pd
import numpy as np


# No. of days for price prediction
pred_days = 100

input_df = price_df.iloc[:X_train_len]
input_df = input_df.iloc[-time_steps:]
date = (input_df.iloc[-1]['Date']).strftime('%Y-%m-%d')
print('Previous Open Day: ', date, '\n')
print(input_df,'\n')
input_array = np.array(input_df['Close']).reshape(-1,1)

output_df = pd.DataFrame({'Date': [], 'Actual Closing Price': [], f'Next {pred_days} days Forecast': []})

# Normalization
input_array = sc.transform(input_array)
# print(type(input_array), input_array.shape)


date_array = pd.to_datetime(price_df['Date'].values).strftime('%Y-%m-%d').values
# Loop over the prediction period and add new rows to the input data for each day
for i in range(pred_days):
  X_input = input_array[-time_steps:]

  # Reshape the data
  X_input_3D = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))

  # Get the scaled predicted price
  output_scaled = model.predict(X_input_3D)

  # add new row to existing input data for prediction
  # print(f"input_array shape: {input_array.shape}, output_scaled shape: {output_scaled.shape}")
  input_array = np.vstack((input_array, output_scaled))

  # Unscaled
  output = sc.inverse_transform(output_scaled)

  # get tmr date
  _, date = get_previous_and_next_open_days(date)
  print(f'Predicted price on {date}: {output[0][0]}')

  # Get closing price on specific date
  if date in date_array:
    # If the search date exists, get its corresponding 'close' column data
    close_data = price_df.loc[price_df['Date'] == date, 'Close'].values[0]
  else:
    close_data = np.nan

  # Create a new row for the current day
  new_row = {'Date': date, 'Actual Closing Price': close_data, f'Next {pred_days} days Forecast': output[0][0]}

  # Append the new row to the input data DataFrame
  output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)

print(output_df)
print(price_df.iloc[X_train_len:])

rmse_forecast_df = output_df.dropna()
print(rmse_forecast_df)

rmse_pred_trend = np.sqrt(((rmse_forecast_df['Actual Closing Price'].values - rmse_forecast_df[f'Next {pred_days} days Forecast'].values) ** 2).mean())
print(f'RMSE for the above days: ', rmse_pred_trend)

pred_output_df = output_df.drop('Actual Closing Price', axis = 1)
interactive_plot(prediction_fig, pred_output_df)
prediction_fig.show()


