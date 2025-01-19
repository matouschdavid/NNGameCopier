from networks_builders.shape_helper import get_shapes
from tensorflow.keras.models import load_model
from io_utils import load_data
from game_utils import encode_frames
from vae import Sampling
from window import Window
import threading

encoder = load_model("models/encoder.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/decoder.keras")
predictor = load_model("models/lstm.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
sequence_length = 120
time_dim = 1
input_prominence = 1
frames, inputs, timestamps = load_data(image_folder, input_file, min=0, max=200)
input_shape, latent_shape = get_shapes(frames)
max_time = 5898 # max time of dataset
frames = frames[-sequence_length:]
inputs = inputs[-sequence_length:]
timestamps = timestamps[-sequence_length:] / max_time
encoder_part, input_part, time_part = encode_frames(encoder, frames, inputs, timestamps, input_prominence)

input_dim = inputs.shape[1]
frame_rate = 15
resolution = (1024, 256)

app = Window()
prediction_thread = threading.Thread(target=app.start_prediction_loop, args=(encoder_part, input_part, time_part, decoder, predictor, input_dim, frame_rate, max_time, input_prominence, time_dim, resolution, latent_shape))
prediction_thread.start()
app.start()

