from tensorflow.keras.models import load_model

from src.constants import NNGCConstants
from src.io_utils import load_data
from src.plot_utils import plot_reconstruction
from src.vae import Sampling, VAE

postfix = NNGCConstants.postfix

encoder_path = f"{NNGCConstants.model_path}vae_encoder{postfix}.keras"
decoder_path = f"{NNGCConstants.model_path}vae_decoder{postfix}.keras"

encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
decoder = load_model(decoder_path)

image_folder = NNGCConstants.image_path
input_file = NNGCConstants.input_file
frames, inputs, _ = load_data(image_folder, input_file, max=500)

vae = VAE(encoder, decoder)
plot_reconstruction(frames, vae, size=10)