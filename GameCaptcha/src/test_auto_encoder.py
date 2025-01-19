from tensorflow.keras.models import load_model
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.plot_utils import plot_reconstruction
from GameCaptcha.src.vae import Sampling, VAE

postfix = "_flappy_64"

encoder_path = f"models/vae_encoder{postfix}.keras"
decoder_path = f"models/vae_decoder{postfix}.keras"

encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
decoder = load_model(decoder_path)

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, _ = load_data(image_folder, input_file, max=500)

vae = VAE(encoder, decoder)
plot_reconstruction(frames, vae, size=10)