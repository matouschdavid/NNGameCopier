import keras

import GameCaptcha.src.config as config

from GameCaptcha.src.networks_builders.decoder import build_decoder
from GameCaptcha.src.networks_builders.encoder import build_encoder
from GameCaptcha.src.networks_builders.vae import VAE
from tensorflow.keras import losses


def build_autoencoder(input_shape, latent_height, latent_width):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_height, latent_width, len(config.frame_channels))
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(), loss=losses.MeanSquaredError())

    return vae, encoder, decoder
