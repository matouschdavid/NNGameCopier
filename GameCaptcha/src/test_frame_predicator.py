from GameCaptcha.src.game_utils import encode_frames
from GameCaptcha.src.plot_utils import plot_prediction
from GameCaptcha.src.io_utils import load_data
from tensorflow import keras
from tensorflow.keras.models import load_model
from GameCaptcha.src.vae import Sampling

keras.config.enable_unsafe_deserialization()
encoder = load_model("models/vae_encoder.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder.keras")
predictor = load_model("models/lstm_model.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
sequence_length = 180
frames, inputs, timestamps = load_data(image_folder, input_file, max=1090)
max_time = 2439
time_dim = 1
input_prominence = 5

frames_slice = frames[-sequence_length:]
inputs_slice = inputs[-sequence_length:]
timestamps_slice = timestamps[-sequence_length:] / max_time

test_sequence = encode_frames(encoder, frames_slice, inputs_slice, timestamps_slice, input_prominence)

jump = [1, 0]
duck = [0, 1]
nothing = [0, 0]
plot_prediction(test_sequence, [jump, duck, nothing], 5, decoder, predictor, max_time, input_prominence, time_dim)