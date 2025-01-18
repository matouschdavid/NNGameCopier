from game_utils import encode_frames
from plot_utils import plot_prediction
from io_utils import load_data
from tensorflow import keras
from tensorflow.keras.models import load_model

from vae import Sampling

keras.config.enable_unsafe_deserialization()
encoder = load_model("models/vae_encoder.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder.keras")
predictor = load_model("models/lstm_model.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
sequence_length = 120
frames, inputs, timestamps = load_data(image_folder, input_file, max=200)
max_time = 5898
time_dim = 1
input_prominence = 5

frames_slice = frames[-sequence_length:]
inputs_slice = inputs[-sequence_length:]
timestamps_slice = timestamps[-sequence_length:] / max_time

test_sequence = encode_frames(encoder, frames_slice, inputs_slice, timestamps_slice, input_prominence)

up = [1, 0, 0, 0]
down = [0, 1, 0, 0]
left = [0, 0, 1, 0]
right = [0, 0, 0, 1]
nothing = [0, 0, 0, 0]
plot_prediction(test_sequence, [up, down, left, right, nothing], 5, decoder, predictor, max_time, input_prominence, time_dim)
