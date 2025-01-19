import config
from tensorflow.keras.models import load_model

from io_utils import load_data
from plot_utils import plot_reconstruction

encoder = load_model(config.encoder_model_path)
decoder = load_model(config.decoder_model_path)

frames, _, _ = load_data(config.compressed_folder, max=500)

plot_reconstruction(frames, encoder, decoder)
