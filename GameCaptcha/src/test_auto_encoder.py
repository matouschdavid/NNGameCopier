from tensorflow.keras.models import load_model
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.plot_utils import plot_reconstruction
from GameCaptcha.src.vae import Sampling, VAE

encoder = load_model("models/vae_encoder_time.keras", custom_objects={"Sampling": Sampling})
decoder = load_model("models/vae_decoder_time.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file, max=500)

vae = VAE(encoder, decoder)
plot_reconstruction(frames, vae, size=100)