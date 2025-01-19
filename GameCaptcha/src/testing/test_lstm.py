import numpy as np

import GameCaptcha.src.config as config
from tensorflow.keras.models import load_model

from GameCaptcha.src.util.io_utils import load_data
from GameCaptcha.src.util.plot_utils import plot_frames, predict_sequence

encoder = load_model(config.encoder_model_path)
decoder = load_model(config.decoder_model_path)
lstm_model = load_model(config.lstm_model_path)

frames, inputs, timestamps = load_data(config.compressed_folder, max=499)
timestamps = timestamps / config.max_time
input_dim = inputs.shape[-1]

inputs_at_start = []
for i in range(input_dim):
    input_at_start = np.zeros(input_dim)
    input_at_start[i] = 1
    inputs_at_start.append(input_at_start)

frames_to_predict = 5

initial_frames = frames[-config.sequence_length:]
input_vectors = inputs[-config.sequence_length:]
time_values = timestamps[-config.sequence_length:]

predicted_frames = predict_sequence(
    encoder, decoder, lstm_model, initial_frames, input_vectors, time_values, frames_to_predict, config.input_prominence, input_dim, inputs_at_start
)

plot_frames(predicted_frames, frames_to_predict + 1)