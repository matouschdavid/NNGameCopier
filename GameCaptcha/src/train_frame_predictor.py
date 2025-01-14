import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input, LSTM, Dropout, BatchNormalization
import matplotlib.pyplot as plt

from GameCaptcha.src.game_utils import predict_next_frame, update_latent_space_buffer, clean_image, \
    remove_input_from_latent_space, encode_frames
from GameCaptcha.src.io_utils import load_data
from tensorflow.keras.models import load_model

from GameCaptcha.src.plot_utils import plot_sequence, plot_prediction
from GameCaptcha.src.vae import Sampling

encoder = load_model("models/vae_encoder_time.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder_time.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file)

max_time = max(timestamps)
print(max_time)

input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]

# latent_dim = int(0.05 * input_height * input_width * input_channels)
latent_dim = 128
input_dim = inputs.shape[1]
print(f"Latent Dimension: {latent_dim}, Input Dimension: {input_dim}")

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length][0:-1])
    return np.array(X), np.array(y)

sequence_length = 60

# Updated latent_dim to include input vector dimension
latent_dim = latent_dim + input_dim

# Input: (sequence_length, latent_dim + 1) to accommodate the time variable
lstm_inputs = keras.Input(shape=(sequence_length, latent_dim + 1))

# Split the input into latent space + input vector and the time variable
main_input = lstm_inputs[..., :-1]  # All except the last feature
time_input = lstm_inputs[..., -1:]  # The last feature is the time variable

# LSTM processing
x = layers.LSTM(128, return_sequences=True)(main_input)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.2)(x)

# Normalize the time variable
normalized_time = layers.BatchNormalization()(time_input)

# Optionally process normalized time separately
time_processed = layers.Dense(32, activation="relu", name="time_dense")(normalized_time)
time_flattened = layers.Flatten(name="flatten_time")(time_processed)
x = layers.Concatenate()([x, time_flattened])

# Final output layer
lstm_outputs = layers.Dense(latent_dim)(x)

# Define and compile the model
lstm_model = Model(lstm_inputs, lstm_outputs, name="lstm_model")
lstm_model.compile(optimizer=keras.optimizers.Adam(), loss="mse")

chunk_size = 2000
for k in range(10):
    for i in range(0, len(frames), chunk_size):
        print(f"({k})Processing {i}:{i + chunk_size}")
        chunk = encode_frames(encoder, frames[i:i + chunk_size], inputs[i:i + chunk_size], timestamps[i:i + chunk_size])
        if len(chunk) > sequence_length:
            X_chunk, y_chunk = create_sequences(chunk, sequence_length)
            lstm_model.fit(X_chunk, y_chunk, epochs=50, batch_size=96, validation_split=0.2)

lstm_model.save("models/lstm_model_time.keras")

test_sequence = encode_frames(encoder, frames[:60], inputs[:60], timestamps[:60])
jump = [1, 0]
duck = [0, 1]
nothing = [0, 0]
plot_prediction(test_sequence, [jump, duck, nothing], 5, decoder, lstm_model)

