import asyncio
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import os
import sys
from fastapi import FastAPI, Request
import uvicorn

# Configure environment variables to accelerate AMD allocations
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import MetaTrader5 as mt5
    from alpaca.data.live import CryptoStream
except ImportError:
    pass

# -------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------------------
SHM_NAME = "alphawicks_market_data"
SHM_SIZE = 48000  # 1000 rows * 6 columns * 8 bytes (float64)
ROW_SIZE = 48     

# -------------------------------------------------------------------
# 1. HARDWARE-ACCELERATED COMPUTE ENGINE (Process B)
# -------------------------------------------------------------------
class RealTimeComputeEngine:
    def __init__(self, pipe_receiver):
        self.pipe = pipe_receiver
        self.shm = None
        self.running = True
        self.data_window_size = 1000
        
        # Verify AMD GPU visibility natively under ROCm via TensorFlow
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if self.gpu_available:
            self.device_name = "/GPU:0"
            print(f"[Engine] AMD ROCm backend active. Bound to GPU: {tf.config.list_physical_devices('GPU')[0].name}")
        else:
            self.device_name = "/CPU:0"
            print("[Engine] AMD GPU not detected via ROCm. Falling back to CPU.")

        # Bind to the zero-copy Shared Memory Matrix
        try:
            self.shm = shared_memory.SharedMemory(name=SHM_NAME)
            self.shared_matrix = np.ndarray((self.data_window_size, 6), dtype=np.float64, buffer=self.shm.buf)
            print("[Engine] Successfully mapped zero-copy view over Shared Memory.")
        except Exception as e:
            print(f"[Engine] Shared memory allocation links failed: {e}")
            self.shared_matrix = None

        # Build the model architecture natively in the target hardware scope
        with tf.device(self.device_name):
            self.model = self.build_lstm_model((60, 1))
            
        self.is_training = False

    def build_lstm_model(self, input_shape):
        """Builds your sequential model. Automatically optimized by AMD MIOpen kernels."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def preprocess_data_rocm(self, np_matrix):
        """
        Replaces your legacy OpenCL C kernel string.
        Calculates min/max transformations natively inside AMD GPU memory.
        """
        with tf.device(self.device_name):
            # 1. Direct generation of tensor arrays in VRAM
            # Pulling only the 'Close' price vector (Column index 4) for this specific LSTM structure
            close_tensor = tf.convert_to_tensor(np_matrix[:, 4], dtype=tf.float32)
            
            # 2. Vectorized range processing inside the AMD compute units
            min_val = tf.reduce_min(close_tensor)
            max_val = tf.reduce_max(close_tensor)
            denom = tf.maximum(max_val - min_val, 1e-8)
            
            # 3. Direct execution of the scale operation without CPU intervention
            scaled_tensor = (close_tensor - min_val) / denom
            return scaled_tensor

    def create_dataset_vram(self, scaled_tensor, time_step=60):
        """Vectorized dataset slicing inside VRAM to eliminate slow Python loops."""
        total_len = tf.shape(scaled_tensor)[0]
        num_samples = total_len - time_step - 1
        
        # Parallel index generation matrix
        idx = tf.range(time_step)[None, :] + tf.range(num_samples)[:, None]
        
        X_train = tf.gather(scaled_tensor, idx)
        y_train = tf.gather(scaled_tensor, tf.range(time_step, total_len - 1))
        
        return tf.reshape(X_train, (num_samples, time_step, 1)), y_train

    async def trigger_background_training(self, scaled_tensor):
        """Runs optimization steps asynchronously so incoming broker data is never blocked."""
        self.is_training = True
        loop = asyncio.get_event_loop()
        
        def run_fit():
            with tf.device(self.device_name):
                X_train, y_train = self.create_dataset_vram(scaled_tensor)
                # Execute training step completely isolated from network tracking threads
                self.model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=0)

        await loop.run_in_executor(None, run_fit)
        self.is_training = False

    async def process_execution_pipeline(self, webhook_signal=None):
        """Coordinates clean tracking transitions across your hardware space."""
        if self.shared_matrix is None:
            return

        if webhook_signal:
            print(f"[Pipeline] Webhook Signal Received: {webhook_signal}")

        # Execute data manipulation entirely in VRAM
        scaled_tensor = self.preprocess_data_rocm(self.shared_matrix)
        
        # Dispatch background updates without causing pipeline lockups
        if not self.is_training:
            asyncio.create_task(self.trigger_background_training(scaled_tensor))
            
        # Squeeze out the exact trailing 60 ticks matrix vector for ultra-fast prediction
        X_test = tf.reshape(scaled_tensor[-60:], (1, 60, 1))
        
        with tf.device(self.device_name):
            prediction = self.model(X_test, training=False)
            # print(f"[AI] Immediate Prediction Output: {prediction.numpy()[0][0]}")

    async def start_broker_streams(self):
        print("[Engine] Booting live WebSocket listening matrices...")
        # Bind your async Alpaca stream or native MT5 handles here

    async def step(self):
        """Asynchronous execution worker that avoids burning an entire core on loop polling."""
        loop = asyncio.get_event_loop()
        if await loop.run_in_executor(None, self.pipe.poll):
            webhook_payload = await loop.run_in_executor(None, self.pipe.recv)
            await self.process_execution_pipeline(webhook_signal=webhook_payload)
        await asyncio.sleep(0.001)

    def close(self):
        if self.shm:
            self.shm.close()

async def run_engine_async_loop(engine):
    await engine.start_broker_streams()
    while engine.running:
        await engine.step()

def start_engine_worker(pipe_receive):
    engine = RealTimeComputeEngine(pipe_receive)
    try:
        asyncio.run(run_engine_async_loop(engine))
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()

# -------------------------------------------------------------------
# 2. LOW-LATENCY NETWORKING & SHM MASTER (Process A)
# -------------------------------------------------------------------
app = FastAPI()
pipe_send = None
master_shm = None

@app.on_event("startup")
async def startup_event():
    global pipe_send, master_shm
    
    try:
        master_shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        base_array = np.ndarray((1000, 6), dtype=np.float64, buffer=master_shm.buf)
        base_array[:] = 0.0
    except FileExistsError:
        master_shm = shared_memory.SharedMemory(name=SHM_NAME)

    pipe_send, pipe_receive = multiprocessing.Pipe(duplex=False)
    
    engine_process = multiprocessing.Process(
        target=start_engine_worker,
        args=(pipe_receive,),
        daemon=True
    )
    engine_process.start()
    print("[Server] High-performance pipeline active on isolated core process.")

@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    pipe_send.send(payload)
    return {"status": "queued_realtime"}

@app.on_event("shutdown")
async def shutdown_event():
    global master_shm
    if master_shm:
        master_shm.close()
        master_shm.unlink()

if __name__ == "__main__":
    uvicorn.run("hpc:app", host="127.0.0.1", port=5000, log_level="warning", workers=1)
