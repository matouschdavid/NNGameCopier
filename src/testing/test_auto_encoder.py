import GameCaptcha.src.config as config
from tensorflow.keras.models import load_model

from GameCaptcha.src.networks_builders.vae import Sampling
from GameCaptcha.src.util.io_utils import load_data
from GameCaptcha.src.util.plot_utils import plot_reconstruction

encoder = load_model(config.encoder_model_path, custom_objects={"Sampling": Sampling})
decoder = load_model(config.decoder_model_path)

frames, _, _ = load_data(config.compressed_folder, max=500)

plot_reconstruction(frames, encoder, decoder, size=10)
