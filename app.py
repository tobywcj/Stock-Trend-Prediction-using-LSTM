import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st



@st.cache_data
def get_data(ticker, start_date):
  tomorrow_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
  raw_price_df = pdr.get_data_yahoo(ticker, start = start_date, end = tomorrow_date)

  return raw_price_df


def plot_data(df, title=''):
  fig = go.Figure()
  for i in df.columns:
    if i != 'Date':
      fig.add_trace(go.Scatter(x = df['Date'], y = df[i], name = i))

  if title: 
    fig.update_layout(title_text = title)

  fig.update_layout(
    xaxis_title = 'Time',
    yaxis_title = 'Close Price USD ($)'
  )

  st.plotly_chart(fig)


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


# convert an array of values into a dataset matrix
def create_dataset(array_data, time_steps=1):

  X, y = [], [] # become a list

  for i in range(time_steps, len(array_data)):
      X.append(array_data[i-time_steps:i, 0]) # its a 2D array => [row, col]
      y.append(array_data[i, 0])

  # return the new dataset
  return np.array(X), np.array(y) # convert into np array for RNN inputs

   
@st.cache_data
def get_pred_price_history(_model, price_df):
   # Prepare the data
    past_data = np.array(price_df['Close']).reshape(-1,1)

    # Scale the data
    sc = MinMaxScaler(feature_range = (0, 1))
    past_data_scaled = sc.fit_transform(past_data)

    # Split the data into train and test sets
    X_pred, _ = create_dataset(past_data_scaled, time_steps)

    # Reshape the data
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1)) # add a new dimensionality to be compatible with the 3D tensor input shape of RNN and allow more indicators

    # Make predictions
    y_pred = _model.predict(X_pred)

    # Unscale the data
    predictions_array = sc.inverse_transform(y_pred)
    predictions_list = predictions_array.reshape(-1).tolist()

    # Create next day prediction column
    price_df = price_df.rename(columns={'Close': 'Actual Closing Price'})

    predictions_list = [np.nan] * time_steps + predictions_list
    price_df['Next Day Prediction'] = predictions_list

    # Display RMSE
    previous_price = np.array(price_df['Actual Closing Price']).reshape(-1,1)
    previous_price = previous_price[time_steps:]
    rmse = np.sqrt(((predictions_array - previous_price) ** 2).mean())

    return price_df, past_data_scaled, sc, rmse


def get_forecast(_model, price_df, past_data_scaled, sc, previous_date):
  one_year_ago = datetime.now() - timedelta(days=365)
  pred_df = price_df[price_df['Date'] > one_year_ago]
  forecast_list = [np.nan] * pred_df.shape[0]
  pred_df[f'Next {pred_days} Days Forecast'] = forecast_list

  forecast_input = past_data_scaled[-time_steps:]

  # Append the dataframe of forecast data to the dataframe of the past stock data
  for _ in range(pred_days):
    X_input = forecast_input[-time_steps:]

    # Reshape the data
    X_input = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))

    # Get the scaled predicted price
    output_scaled = _model.predict(X_input)

    # add new row to existing input data for prediction
    forecast_input = np.vstack((forecast_input, output_scaled))

    # Unscaled
    output = sc.inverse_transform(output_scaled)

    # get tmr date
    _, previous_date = get_previous_and_next_open_days(previous_date)
    print(f'Predicted price on {previous_date}: {output[0][0]}')

    # Create a new row for the current day
    new_row = {'Date': previous_date, 
              'Actual Closing Price': np.nan,
              'Next Day Prediction': np.nan, 
              f'Next {pred_days} Days Forecast': output[0][0]}

    # Append the new row to the input data DataFrame
    pred_df = pd.concat([pred_df, pd.DataFrame([new_row])], ignore_index=True)

  pred_df.loc[pred_df['Date'] == next_open_day, 'Next Day Prediction'] = pred_df.loc[pred_df['Date'] == next_open_day, f'Next {pred_days} Days Forecast']
  
  return pred_df



if __name__ == "__main__":
      
  ############################################################ MAIN PAGE widgets ############################################################

  st.title("📈 Stock Trend Prediction")
  st.markdown("""This app provides a web application interface for users to select a stock ticker and visualize the next-day stock price prediction, 
              as well as forecast the future trends using Long Short-Term Memory (LSTM) neural networks 🧠""")

  st.divider()

  yf.pdr_override() # download data faster

  time_steps = 60

  # download dataframe
  ticker = st.selectbox(
    'Select a stock ticker',
    ('AAPL', 'TSLA', 'GOOGL', 'AMZN', 'JPM', 'IVV', 'META', 'NVDA', 'BRK-B'))


  if ticker:
    model = load_model(f'models/{ticker}_model.h5')

    with st.spinner(f'Loading {ticker} stock data from Yahoo Finance ...'):
      raw_price_df = get_data(ticker, '2010-01-01')
      price_df = raw_price_df.reset_index()[['Date', 'Close']]
      previous_date = (price_df.iloc[-1]['Date']).strftime('%Y-%m-%d')
      _, next_open_day = get_previous_and_next_open_days(previous_date)

    # describe dataframe
    st.subheader(f'Data from 2010-01-01 to {previous_date}')
    st.write(raw_price_df.describe())

    st.write(f'Previous Open Day:  {previous_date}')
    st.write(f'Next Open Day:  {next_open_day}')


    st.divider()


    st.subheader(f'{ticker} Stock Price History')

    with st.spinner(f'Predicting {ticker} Stock Price ...'):
      price_df, past_data_scaled, sc, rmse = get_pred_price_history(model, price_df)
      plot_data(price_df)
      st.write(f"Root Mean Squared Error on Previous Stock Price: ${rmse} ")
    

    st.divider()


    st.subheader('Stock Trend Forecast')

    # Forecasting market trend
    pred_days = st.slider('Days of Prediction', 0, 100, 10, 5)

    if st.button('Predict'):

      with st.spinner(f'Forecasting {pred_days}-day Trend...'):
        pred_df = get_forecast(model, price_df, past_data_scaled, sc, previous_date)

      # Plot a graph of actual closing price vs predicted price
      plot_data(pred_df, f'Next {pred_days} Days Forecast')

      st.dataframe(pred_df.tail(pred_days).drop(columns='Actual Closing Price').set_index('Date'))
