from tensorflow.keras.models import load_model
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.game_utils import encode_frames
from GameCaptcha.src.vae import Sampling
from GameCaptcha.src.window import Window
import threading

encoder = load_model("models/vae_encoder_flappy.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder_flappy.keras")
predictor = load_model("models/lstm_model_flappy.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"


sequence_length = 180
frames, inputs, timestamps = load_data(image_folder, input_file, min=0, max=sequence_length)
frames = frames[-sequence_length:]
inputs = inputs[-sequence_length:]
latent_space_buffer = encode_frames(encoder, frames, inputs)

input_dim = inputs.shape[1]
frame_rate = 30

app = Window()
prediction_thread = threading.Thread(target=app.start_prediction_loop, args=(latent_space_buffer, decoder, predictor, input_dim, frame_rate))
prediction_thread.start()
app.start()

