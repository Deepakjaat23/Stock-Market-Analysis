import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import itertools
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/deepa/OneDrive/Desktop/cleaned_stock_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
companies = df['Company'].unique()

# Sidebar
st.sidebar.title("Stock Forecasting App")
selected_company = st.sidebar.selectbox("Select Company", companies)
selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
forecast_days = st.sidebar.slider("Days to Forecast", 10, 90, 30)

# Filter data
company_df = df[df['Company'] == selected_company].set_index('Date')
company_close = company_df['Close'].asfreq('D').fillna(method='ffill')

# Tabs
eda_tab, forecast_tab = st.tabs(["📊 EDA Dashboard", "📈 Forecasting"])

# ---------- EDA Dashboard ---------- #
with eda_tab:
    st.subheader(f"EDA for {selected_company}")
    
    # Plot Close Price
    st.line_chart(company_df['Close'])

    # Moving Averages
    company_df['MA_20'] = company_df['Close'].rolling(20).mean()
    company_df['MA_50'] = company_df['Close'].rolling(50).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(company_df['Close'], label='Close')
    ax.plot(company_df['MA_20'], label='MA 20')
    ax.plot(company_df['MA_50'], label='MA 50')
    ax.legend()
    st.pyplot(fig)

    # Volatility
    company_df['Volatility'] = company_df['Close'].pct_change().rolling(20).std()
    st.line_chart(company_df['Volatility'])

# ---------- Forecasting Tab ---------- #
with forecast_tab:
    st.subheader(f"{selected_model} Forecast for {selected_company}")

    if selected_model == "ARIMA":
        best_aic = np.inf
        best_order = None
        for p, d, q in itertools.product(range(0, 3), repeat=3):
            try:
                model = ARIMA(company_close, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue
        model_arima = ARIMA(company_close, order=best_order)
        fit_arima = model_arima.fit()
        forecast = fit_arima.forecast(steps=forecast_days)
        st.line_chart(pd.concat([company_close, forecast]))

    elif selected_model == "SARIMA":
        best_aic = np.inf
        for order in itertools.product(range(0, 3), repeat=3):
            for seasonal in itertools.product(range(0, 2), repeat=3):
                try:
                    model = SARIMAX(company_close, order=order, seasonal_order=seasonal + (12,))
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = order
                        best_seasonal = seasonal
                except:
                    continue
        model_sarima = SARIMAX(company_close, order=best_order, seasonal_order=best_seasonal + (12,))
        fit_sarima = model_sarima.fit(disp=False)
        forecast = fit_sarima.forecast(steps=forecast_days)
        st.line_chart(pd.concat([company_close, forecast]))

    elif selected_model == "Prophet":
        prophet_df = company_close.reset_index()
        prophet_df.columns = ['ds', 'y']
        model_prophet = Prophet(daily_seasonality=True)
        model_prophet.fit(prophet_df)
        future = model_prophet.make_future_dataframe(periods=forecast_days)
        forecast = model_prophet.predict(future)
        fig = model_prophet.plot(forecast)
        st.pyplot(fig)

    elif selected_model == "LSTM":
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(company_close.values.reshape(-1, 1))

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)

        SEQ_LEN = 60
        X, y = create_sequences(scaled_close, SEQ_LEN)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_lstm = Sequential()
        model_lstm.add(LSTM(units=50, return_sequences=False, input_shape=(SEQ_LEN, 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        last_seq = scaled_close[-SEQ_LEN:]
        input_seq = last_seq.reshape(1, SEQ_LEN, 1)
        forecast_scaled = []
        for _ in range(forecast_days):
            pred = model_lstm.predict(input_seq)[0][0]
            forecast_scaled.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        future_index = pd.date_range(start=company_close.index[-1]+pd.Timedelta(days=1), periods=forecast_days)
        forecast_series = pd.Series(forecast, index=future_index)
        st.line_chart(pd.concat([company_close, forecast_series]))
