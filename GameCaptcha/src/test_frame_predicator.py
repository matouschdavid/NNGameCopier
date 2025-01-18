from GameCaptcha.src.game_utils import encode_frames
from GameCaptcha.src.plot_utils import plot_prediction
from GameCaptcha.src.io_utils import load_data
from tensorflow import keras
from tensorflow.keras.models import load_model

from GameCaptcha.src.vae import Sampling

keras.config.enable_unsafe_deserialization()
encoder = load_model("models/vae_encoder_flappy.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder_flappy.keras")
predictor = load_model("models/lstm_model_flappy.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, _ = load_data(image_folder, input_file, min=0, max=180)
last_frame = 180
first_mem_frame = last_frame - 180

frames_slice = frames[first_mem_frame:last_frame]
inputs_slice = inputs[first_mem_frame:last_frame]

test_sequence = encode_frames(encoder, frames_slice, inputs_slice)
print(test_sequence[-1])

flap = [1]
nothing = [0]
plot_prediction(test_sequence, [flap, nothing], 5, decoder, predictor)