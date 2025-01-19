from tensorflow.keras.models import load_model

from io_utils import load_data
from plot_utils import plot_reconstruction

encoder = load_model("models/encoder.keras")
decoder = load_model("models/decoder.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, _, _ = load_data(image_folder, input_file, max=500)

plot_reconstruction(frames, encoder, decoder)
