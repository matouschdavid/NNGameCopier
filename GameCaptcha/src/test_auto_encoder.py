from tensorflow.keras.models import load_model
from io_utils import load_data
from plot_utils import plot_reconstruction
from vae import Sampling, VAE

encoder = load_model("models/vae_encoder.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file, max=500)

vae = VAE(encoder, decoder)
plot_reconstruction(frames, vae, size=100)