from keras import Model

from io_utils import load_data
from networks_builders.auto_encoder import build_autoencoder
from plot_utils import plot_loss, plot_reconstruction
import os

os.environ['HSA_OVERRIDE_GFX_VERSION']="11.0.0"

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, _, _ = load_data(image_folder, input_file)

input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]
input_shape = (input_height, input_width, input_channels) # 48 x 256 x 1
downscale_factor = 16
latent_height = int(input_height / downscale_factor)
latent_width = int(input_width / downscale_factor)
latent_channels = 64


autoencoder, encoder, decoder = build_autoencoder(input_shape, latent_height, latent_width, latent_channels)
history = autoencoder.fit(frames, frames, batch_size=128, epochs=100, validation_split=0.2)

print("Done training")
encoder.save("models/encoder.keras")
decoder.save("models/decoder.keras")
print("Models saved")
plot_loss(history)
plot_reconstruction(frames, encoder, decoder)
