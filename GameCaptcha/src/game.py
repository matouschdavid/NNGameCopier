from tensorflow.keras.models import load_model
from GameCaptcha.src.game_utils import encode_frames
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.vae import Sampling
from GameCaptcha.src.window import Window
import threading

encoder = load_model("models/vae_encoder.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder.keras")
predictor = load_model("models/lstm_model.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs = load_data(image_folder, input_file, max=110)
frames = frames[-60:]
inputs = inputs[-60:]
latent_space_buffer = encode_frames(encoder, frames, inputs)

input_dim = inputs.shape[1]
frame_rate = 15

app = Window()
prediction_thread = threading.Thread(target=app.start_prediction_loop, args=(latent_space_buffer, decoder, predictor, input_dim, frame_rate))
prediction_thread.start()
app.start()

