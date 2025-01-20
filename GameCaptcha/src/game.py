from tensorflow.keras.models import load_model

from GameCaptcha.src.networks_builders.vae import Sampling
from GameCaptcha.src.util.io_utils import load_data
from GameCaptcha.src.util.game_utils import encode_frames
from GameCaptcha.src.util.window import Window
import threading
import GameCaptcha.src.config as config

encoder = load_model(config.encoder_model_path, custom_objects={"Sampling": Sampling})
decoder = load_model(config.decoder_model_path)
lstm = load_model(config.lstm_model_path)

frames, inputs, timestamps = load_data(config.compressed_folder, min=0, max=200)
frames = frames[-config.sequence_length:]
inputs = inputs[-config.sequence_length:]
timestamps = timestamps[-config.sequence_length:] / config.max_time
encoder_part, input_part, time_part = encode_frames(encoder, frames, inputs, timestamps)

app = Window()
prediction_thread = threading.Thread(target=app.start_prediction_loop, args=(encoder_part, input_part, time_part, decoder, lstm))
prediction_thread.start()
app.start()

