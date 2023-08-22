# 📈 Stock Price Predictions Using LSTM
This project aims to predict stock prices and trends using Long Short-Term Memory (LSTM) neural networks 🧠

It is implemented in Python using Keras and Tensorflow.

## 📊 Overview
The model is trained on historical stock price data downloaded from Yahoo Finance using yfinance.

Features like closing price are used to train a LSTM model for each stock.

The trained model takes in a number of previous closing prices to predict the next day's closing price.

This is extended to forecast the price trend over a number of days into the future.

## 🚀 Usage
The trained models are loaded and used within a Streamlit web app for interactive stock price prediction.

Users can select a stock, view historical data, and get next-day predictions as well as a forecast trend.

The app is deployed on Streamlit Cloud for easy access: https://share.streamlit.io/... ↗

The repository contains:

app.py - Streamlit app code
Stock Price Predictions Using LSTM.ipynb - Jupyter notebook for model training
Model files - containing trained LSTM models for each stock

## 🎯 Goal
The aim is to explore LSTM for stock price time series forecasting.

The interactive web app allows easy use and demonstration of the capabilities of the LSTM models.

## 🧠 Learnings
How to format time series data for training LSTMs
Tuning LSTM neural networks for forecasting tasks
Deploying ML models via web apps for easy user access
Overall this project provided useful experience in applying deep learning to stock market prediction!
