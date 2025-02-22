import GameCaptcha.src.config as config
from GameCaptcha.src.networks_builders.auto_encoder import build_autoencoder
from GameCaptcha.src.util.io_utils import load_data
from GameCaptcha.src.util.plot_utils import plot_loss, plot_reconstruction

frames, _, _ = load_data(config.compressed_folder)

input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]
input_shape = (input_height, input_width, input_channels)
latent_height = int(input_height / config.downscale_factor)
latent_width = int(input_width / config.downscale_factor)

autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_height, latent_width)
history = autoencoder.fit(frames, batch_size=128, epochs=50)

print("Done training")
encoder.save(config.encoder_model_path)
decoder.save(config.decoder_model_path)
print("Models saved")
plot_loss(history)
plot_reconstruction(frames, encoder, decoder)
