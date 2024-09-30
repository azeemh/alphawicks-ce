import numpy as np
import talib
import ccxt
import alpaca_trade_api as tradeapi
import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pyopencl as cl #pip install pyopencl 

# Alpaca API setup
ALPACA_API_KEY = 'your-alpaca-api-key'
ALPACA_API_SECRET = 'your-alpaca-api-secret'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, api_version='v2')

# MetaTrader 5 setup
mt5.initialize()

# OpenCL context setup
def opencl_setup():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    return context, queue

# OpenCL-based data preprocessing
def preprocess_data_opencl(data, context, queue):
    mf = cl.mem_flags
    data_np = np.array(data, dtype=np.float32)
    data_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np)
    
    program = cl.Program(context, """
    __kernel void scale(__global const float* input, __global float* output, float min_val, float max_val) {
        int gid = get_global_id(0);
        output[gid] = (input[gid] - min_val) / (max_val - min_val);
    }
    """).build()
    
    output_np = np.zeros_like(data_np)
    output_buffer = cl.Buffer(context, mf.WRITE_ONLY, output_np.nbytes)
    
    min_val, max_val = data_np.min(), data_np.max()
    program.scale(queue, data_np.shape, None, data_buffer, output_buffer, np.float32(min_val), np.float32(max_val))
    
    cl.enqueue_copy(queue, output_np, output_buffer).wait()
    return output_np

# Real-time scaler for incremental data
class RealTimeScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False
    
    def scale_incremental(self, data):
        if not self.scaler_fitted:
            self.scaler.fit(data.reshape(-1, 1))
            self.scaler_fitted = True
        return self.scaler.transform(data.reshape(-1, 1))

# LSTM model building function
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Dataset creation for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# LSTM real-time training with OpenCL
def train_lstm_realtime(data, model, scaler, time_step=60):
    scaled_data = scaler.scale_incremental(data)
    X_train, y_train = create_dataset(scaled_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0)
    return model

# Real-time market data streaming from Alpaca
def stream_market_data_alpaca(symbol, timeframe='minute'):
    while True:
        barset = alpaca_api.get_barset(symbol, timeframe, limit=1)
        yield barset[symbol][0]

# Real-time market data streaming from MetaTrader 5
def stream_market_data_mt5(symbol, timeframe=mt5.TIMEFRAME_M1):
    while True:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        yield rates[0]

# Main function to run the trading system
def main():
    symbol = 'BTC/USD'
    
    # OpenCL context setup
    context, queue = opencl_setup()
    
    # Initialize the real-time scaler
    scaler = RealTimeScaler()
    
    # Build the LSTM model
    model = build_lstm_model((60, 1))  # 60 time steps
    
    # Streaming real-time data from Alpaca
    market_stream = stream_market_data_alpaca(symbol)
    
    data_window = []
    
    # Main loop for real-time data processing
    for new_bar in market_stream:
        close_price = new_bar.c  # Close price
        data_window.append(close_price)

        if len(data_window) >= 60:
            # Preprocess data using OpenCL
            preprocessed_data = preprocess_data_opencl(np.array(data_window), context, queue)
            
            # Train the LSTM model in real-time
            model = train_lstm_realtime(preprocessed_data, model, scaler)
            
            # Remove oldest data point
            data_window.pop(0)
        
        # Once 60 data points are available, make a prediction
        if len(data_window) == 60:
            X_test = np.reshape(preprocessed_data, (1, 60, 1))
            prediction = model.predict(X_test)
            print(f"Predicted next price: {prediction[0][0]}")

if __name__ == "__main__":
    main()
