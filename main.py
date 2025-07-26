from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

app = FastAPI()

class PredictionRequest(BaseModel):
    training_days: int = Query(..., ge=30, le=365)
    prediction_days: int = Query(..., ge=1, le=60)
    neurons: int = Query(..., ge=10, le=200)
    window_size: int = Query(30, ge=7, le=90)

class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence: float
    mse: float

@app.post("/predict", response_model=PredictionResponse)
def predict_lstm(request: PredictionRequest):
    def fetch_btc_prices(days):
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        res = requests.get(url, params=params).json()
        if "prices" not in res:
            raise ValueError("Error fetching price data from API")
        df = pd.DataFrame(res["prices"], columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["date", "price"]]

    def add_features(df):
        df["ma7"] = df["price"].rolling(7).mean()
        df["ema7"] = df["price"].ewm(span=7, adjust=False).mean()
        df["pct_change"] = df["price"].pct_change()
        df["volatility"] = df["price"].rolling(7).std()
        return df.dropna().reset_index(drop=True)

    def prepare_data(df, feature_cols, window_size, prediction_days):
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        X, y = [], []
        for i in range(window_size, len(scaled) - prediction_days + 1):
            X.append(scaled[i - window_size:i])
            y.append(scaled[i:i + prediction_days, 0])
        return np.array(X), np.array(y), scaler

    def build_model(input_shape, output_steps, neurons):
        model = Sequential([
            LSTM(neurons, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(neurons),
            Dropout(0.2),
            Dense(output_steps)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_future(model, df, scaler, feature_cols, window_size, prediction_days):
        last_window = df[feature_cols].values[-window_size:]
        last_scaled = scaler.transform(last_window)
        X_input = last_scaled.reshape(1, window_size, len(feature_cols))
        pred_scaled = model.predict(X_input)
        padded = np.zeros((prediction_days, len(feature_cols)))
        padded[:, 0] = pred_scaled[0]
        inv = scaler.inverse_transform(padded)
        return inv[:, 0]

    df = fetch_btc_prices(request.training_days)
    df = add_features(df)
    feature_cols = ["price", "ma7", "ema7", "pct_change", "volatility"]
    X, y, scaler = prepare_data(df, feature_cols, request.window_size, request.prediction_days)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(input_shape=(request.window_size, len(feature_cols)),
                        output_steps=request.prediction_days,
                        neurons=request.neurons)
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_data=(X_val, y_val))

    val_preds = model.predict(X_val)
    val_mse = mean_squared_error(y_val.flatten(), val_preds.flatten())
    confidence = max(65, min(95, 100 - val_mse * 1000))
    predictions = predict_future(model, df, scaler, feature_cols, request.window_size, request.prediction_days)

    return PredictionResponse(predictions=predictions.tolist(), confidence=confidence, mse=float(val_mse))
