# Stock Market Forecasting using Time Series Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import itertools
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv('C:/Users/deepa/OneDrive/Desktop/stock_details_5_years.csv/stock_details_5_years.csv')
df['Date'] = pd.to_datetime(df['Date'].str.split(' ').str[0])
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
df = df[df['Company'] == 'AAPL'].set_index('Date')
aapl_close = df['Close'].asfreq('D').fillna(method='ffill')

# --- ARIMA (Tuned) ---
from statsmodels.tsa.arima.model import ARIMA

best_aic = np.inf
best_order = None
for p, d, q in itertools.product(range(0, 3), repeat=3):
    try:
        model = ARIMA(aapl_close, order=(p, d, q))
        result = model.fit()
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = (p, d, q)
    except:
        continue

print(f"Best ARIMA order: {best_order}")
model_arima = ARIMA(aapl_close, order=best_order)
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=30)

plt.figure(figsize=(12, 5))
plt.plot(aapl_close, label='Historical')
plt.plot(forecast_arima.index, forecast_arima, label='ARIMA Forecast', color='red')
plt.title('ARIMA Forecast - AAPL')
plt.legend()
plt.grid(True)
plt.show()

# --- SARIMA (Tuned) ---
from statsmodels.tsa.statespace.sarimax import SARIMAX

best_aic_sarima = np.inf
best_order_sarima = None
best_seasonal_order = None
for order in itertools.product(range(0, 3), repeat=3):
    for seasonal_order in itertools.product(range(0, 2), repeat=3):
        try:
            model = SARIMAX(aapl_close, order=order, seasonal_order=seasonal_order + (12,))
            results = model.fit(disp=False)
            if results.aic < best_aic_sarima:
                best_aic_sarima = results.aic
                best_order_sarima = order
                best_seasonal_order = seasonal_order
        except:
            continue

print(f"Best SARIMA order: {best_order_sarima}, seasonal_order: {best_seasonal_order + (12,)}")
model_sarima = SARIMAX(aapl_close, order=best_order_sarima, seasonal_order=best_seasonal_order + (12,))
fit_sarima = model_sarima.fit(disp=False)
forecast_sarima = fit_sarima.forecast(steps=30)

plt.figure(figsize=(12, 5))
plt.plot(aapl_close, label='Historical')
plt.plot(forecast_sarima.index, forecast_sarima, label='SARIMA Forecast', color='green')
plt.title('SARIMA Forecast - AAPL')
plt.legend()
plt.grid(True)
plt.show()

# --- Prophet ---
from prophet import Prophet
prophet_df = aapl_close.reset_index()
prophet_df.columns = ['ds', 'y']
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

model_prophet.plot(forecast_prophet)
plt.title("Prophet Forecast - AAPL")
plt.show()

# --- LSTM ---
from keras.models import Sequential
from keras.layers import LSTM, Dense

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(aapl_close.values.reshape(-1, 1))

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
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

predicted = model_lstm.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(aapl_close.index[-len(predicted):], y_actual, label='Actual')
plt.plot(aapl_close.index[-len(predicted):], predicted, label='LSTM Prediction', color='orange')
plt.title('LSTM Forecast - AAPL')
plt.legend()
plt.grid(True)
plt.show()

# --- Evaluation ---
rmse_arima = sqrt(mean_squared_error(aapl_close[-30:], forecast_arima))
rmse_sarima = sqrt(mean_squared_error(aapl_close[-30:], forecast_sarima))
rmse_lstm = sqrt(mean_squared_error(y_actual, predicted))

# Prophet RMSE evaluation (last 30 days only)
prophet_pred = forecast_prophet[['ds', 'yhat']].set_index('ds').join(aapl_close)
prophet_pred.dropna(inplace=True)
rmse_prophet = sqrt(mean_squared_error(prophet_pred['Close'], prophet_pred['yhat']))

# Print RMSEs
print("\nModel RMSE Comparison:")
print(f"Best ARIMA {best_order} RMSE: {rmse_arima:.2f}")
print(f"Best SARIMA {best_order_sarima}, seasonal={best_seasonal_order + (12,)} RMSE: {rmse_sarima:.2f}")
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
