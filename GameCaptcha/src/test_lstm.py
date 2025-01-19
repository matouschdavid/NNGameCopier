import config
from tensorflow.keras.models import load_model

from io_utils import load_data
from networks_builders.shape_helper import get_shapes

from plot_utils import predict_sequence, plot_frames
import numpy as np

encoder = load_model(config.encoder_model_path)
decoder = load_model(config.decoder_model_path)
lstm_model = load_model(config.lstm_model_path)

frames, inputs, timestamps = load_data(config.compressed_folder, max=1121)
timestamps = timestamps / config.max_time
input_shape, latent_shape = get_shapes(frames)
input_dim = inputs.shape[-1]

up = [1, 0, 0, 0]
down = [0, 1, 0, 0]
left = [0, 0, 1, 0]
right = [0, 0, 0, 1]
nothing = [0, 0, 0, 0]
inputs_at_start = [up, down, left, right, nothing]
frames_to_predict = 5

initial_frames = frames[-config.sequence_length:]
plot_frames(initial_frames)
input_vectors = inputs[-config.sequence_length:]
time_values = timestamps[-config.sequence_length:]

predicted_frames = predict_sequence(
    encoder, decoder, lstm_model, initial_frames, input_vectors, time_values, frames_to_predict, 1, input_dim, inputs_at_start
) # todo add prominence

# Visualize the predicted frames
plot_frames(predicted_frames)