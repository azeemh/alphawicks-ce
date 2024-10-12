import numpy as np
import pandas as pd
import talib
import ccxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from pyopencl import cl, cl_array

# OpenCL context setup
def opencl_setup():
    platform = cl.get_platforms()[0]  # Assuming single platform for simplicity
    device = platform.get_devices()[0]  # Assuming single device for simplicity
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    return context, queue

def preprocess_data_opencl(data, context, queue):
    """Use OpenCL for parallelized data processing."""
    mf = cl.mem_flags
    data_np = np.array(data, dtype=np.float32)
    data_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np)
    
    # Define OpenCL kernel for processing (this is a simplified example)
    program = cl.Program(context, """
    __kernel void scale(__global const float* input, __global float* output, float min_val, float max_val) {
        int gid = get_global_id(0);
        output[gid] = (input[gid] - min_val) / (max_val - min_val);
    }
    """).build()
    
    output_np = np.zeros_like(data_np)
    output_buffer = cl.Buffer(context, mf.WRITE_ONLY, output_np.nbytes)
    
    # Assuming min/max values are known for real-time scaling
    min_val, max_val = data_np.min(), data_np.max()
    program.scale(queue, data_np.shape, None, data_buffer, output_buffer, np.float32(min_val), np.float32(max_val))
    
    cl.enqueue_copy(queue, output_np, output_buffer).wait()
    return output_np

# Preprocess real-time data incrementally
class RealTimeScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False
    
    def scale_incremental(self, data):
        if not self.scaler_fitted:
            self.scaler.fit(data.reshape(-1, 1))
            self.scaler_fitted = True
        return self.scaler.transform(data.reshape(-1, 1))

# LSTM Model Building & Real-Time Prediction
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))  # Output layer for prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(data, time_step=60):
    """Create real-time dataset from incoming data using rolling windows."""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_lstm_realtime(data, model, scaler, time_step=60):
    scaled_data = scaler.scale_incremental(data)
    X_train, y_train = create_dataset(scaled_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0)
    return model

# Real-Time Market Data Streaming
def stream_market_data(exchange_id, symbol, timeframe='1m'):
    exchange_class = getattr(ccxt, exchange_id)  # Create exchange instance dynamically
    exchange = exchange_class({'enableRateLimit': True})

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
        yield ohlcv[-1]  # Return latest OHLCV candle

# Main Real-Time Loop
def main():
    exchange_id = 'bitstamp'
    symbol = 'BTC/USD'
    
    context, queue = opencl_setup()
    scaler = RealTimeScaler()
    
    # Initialize LSTM model
    model = build_lstm_model((60, 1))  # 60 time steps

    # Streaming real-time data
    market_stream = stream_market_data(exchange_id, symbol)

    data_window = []
    
    for new_candle in market_stream:
        close_price = new_candle[4]  # Close price
        data_window.append(close_price)

        if len(data_window) >= 60:  # Minimum window size for LSTM
            # Parallelized preprocessing using OpenCL
            preprocessed_data = preprocess_data_opencl(np.array(data_window), context, queue)
            model = train_lstm_realtime(preprocessed_data, model, scaler)
            data_window.pop(0)  # Slide the window forward
        
        # Model can now predict using the last 'data_window'
        if len(data_window) == 60:
            X_test = np.reshape(preprocessed_data, (1, 60, 1))
            prediction = model.predict(X_test)
            print(f"Predicted next price: {prediction[0][0]}")
            
if __name__ == "__main__":
    main()
    
    