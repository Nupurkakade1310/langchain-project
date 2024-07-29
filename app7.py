import streamlit as st
import yfinance as yf # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import matplotlib.pyplot as plt

# Function to fetch and process stock data
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

# Function to build a simple linear regression model
def train_model(df):
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
    X = df[['Date']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit interface
st.title('Stock Market Prediction Tool')

ticker = st.text_input('Enter stock ticker:', 'AAPL')
start_date = st.date_input('Start date:', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date:', pd.to_datetime('2023-01-01'))

if st.button('Predict'):
    df = get_stock_data(ticker, start_date, end_date)
    st.write(df.head())

    model, X_test, y_test = train_model(df)

    # Making predictions
    predictions = model.predict(X_test)
    st.write("Model performance:")
    st.write(f"Mean squared error: {((y_test - predictions) ** 2).mean()}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Actual Prices')
    plt.plot(X_test, predictions, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Stock Prices for {ticker}')
    plt.legend()
    st.pyplot(plt)
