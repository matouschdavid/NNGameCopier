import config
from keras import Model

from io_utils import load_data
from networks_builders.auto_encoder import build_autoencoder
from plot_utils import plot_loss, plot_reconstruction

frames, _, _ = load_data(config.compressed_folder)

input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]
input_shape = (input_height, input_width, input_channels)
latent_height = int(input_height / config.downscale_factor)
latent_width = int(input_width / config.downscale_factor)

autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_height, latent_width)
history = autoencoder.fit(frames, frames, batch_size=128, epochs=100, validation_split=0.2)

print("Done training")
encoder.save(config.encoder_model_path)
decoder.save(config.decoder_model_path)
print("Models saved")
plot_loss(history)
plot_reconstruction(frames, encoder, decoder)
