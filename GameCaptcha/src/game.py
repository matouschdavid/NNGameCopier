from tensorflow.keras.models import load_model
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.game_utils import encode_frames
from GameCaptcha.src.vae import Sampling
from GameCaptcha.src.window import Window
import threading

encoder = load_model("models/vae_encoder_tiny.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder_tiny.keras")
predictor = load_model("models/lstm_model_tiny.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file, min=0, max=120)
frames = frames[-120:]
inputs = inputs[-120:]
latent_space_buffer = encode_frames(encoder, frames, inputs, timestamps)

input_dim = inputs.shape[1]
frame_rate = 15

app = Window()
prediction_thread = threading.Thread(target=app.start_prediction_loop, args=(latent_space_buffer, decoder, predictor, input_dim, frame_rate))
prediction_thread.start()
app.start()

