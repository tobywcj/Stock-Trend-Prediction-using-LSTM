# 📈 Stock Price Predictions Using LSTM
This project aims to predict next-day stock price and trends using Long Short-Term Memory (LSTM) neural networks 🧠

It includes two main components: the training of the LSTM model using the 'Stock Price Prediction Using LSTM.ipynb' script and the deployment of the prediction application using the 'app.py' script on Streamlit Cloud.


## 📊 Overview
The model is trained on historical stock price data downloaded from Yahoo Finance using yfinance.

Features like closing price are used to train a LSTM model for each stock.

The trained model takes in a number of previous closing prices to predict the next day's closing price.

This is extended to forecast the price trend over a user-adjusted number of days into the future.
 

## 🚀 Usage
- The trained models are loaded and used within a Streamlit web app for interactive stock price prediction.

- Users can select a stock, view model-predicted stock price history, and get next-day predictions as well as a forecast trend.

- The app is deployed on Streamlit Cloud for easy access and FREE: 
    [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-trend-prediction-lstm-analysis.streamlit.app/)


## 📦 This Repository contains:
- app.py - Streamlit app code

- Stock Price Predictions Using LSTM.ipynb - Jupyter notebook for training a LSTM model of a new stock

- Model files - containing trained LSTM models for each stock


## 🚀 How to train a new stock model locally (Google Drive):
1. Download this repository to your local machine. 📥

2. Open the terminal or command prompt and navigate to the directory where you downloaded the folder. 💻

3. Create a Python virtual environment for the project. Run the following commands in the terminal: 🛠️
    ```sh
    python -m venv [project_venv]
    .\[project_venv]\Scripts\activate (Windows)
    source ./[project_venv]/bin/activate (macOS/Linux)
    pip install -r requirements.txt
    ```

4. Run the 'Stock Price Prediction Using LSTM.ipynb' script on Google Colab to train a LSTM model on historical stock price data of the new ticker user entered.

5. New stock model should be appeared in the 'model/' folder as [new_stock_ticker]_model.h5

6. Append the new stock ticker name to the select list
    ```sh
    ticker = st.selectbox(
        'Select a stock ticker',
        ('AAPL', 'TSLA', 'GOOGL', 'AMZN', 'JPM', 'IVV', 'META', 'NVDA', 'BRK-B', '[new_ticker]'))
    ```
7. Start the Streamlit app by running the following command in the terminal: 💭
    ```sh
    streamlit run app.py
    ```

8. Explore the stock price history, view the price prediction for the next day, and forecast the trend for a specific number of days.

9. Enjoy gaining insights into stock market trends and making informed decisions.


## 🎯 Goal
- The aim is to explore LSTM for stock price time series forecasting. 📈💡

- The interactive web app allows easy use and demonstration of the capabilities of the LSTM models. 🌐🚀


## ⚠️ Pitfalls
- The model learns that the next closing price wouldn't be far from the current closing price, leading to predictions that are close to the current price. 📉📈


## 💡 Note:
- Stock market price is inherently a random walk. An LSTM, let alone any kind of machine learning model, can’t predict something that is inherently random.(sometimes even a monkey can outperform a fund manager!) 📉🧠

- The project is solely for the educational value, which is a good way to learn tackling time series data using LSTM. 📊🔍

- The LSTM model is trained on historical data, and its predictions and forecasts should be treated as informative indicators rather than guaranteed predictions. 📉🔮

- Modifications to the code, such as adding technical indicators and improving hyperparameters, are suggested to enhance the model's performance. 🛠️✨



## 🧠 Learnings
- How to format time series data for training LSTMs. 📚📈

- Tuning LSTM neural networks for forecasting tasks. 🔄🧠

- Deploying ML models via web apps for easy user access. 🌐💻

- Overall, this project provided useful experience in applying deep learning to stock market prediction! 📈🧠💡
