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
frames, inputs, _ = load_data(image_folder, input_file, min=0, max=120)
last_frame = 120
first_mem_frame = last_frame - 120

frames_slice = frames[first_mem_frame:last_frame]
inputs_slice = inputs[first_mem_frame:last_frame]

test_sequence = encode_frames(encoder, frames_slice, inputs_slice)
print(test_sequence[-1])

jump = [1, 0]
duck = [0, 1]
nothing = [0, 0]
plot_prediction(test_sequence, [jump, duck, nothing], 5, decoder, predictor)