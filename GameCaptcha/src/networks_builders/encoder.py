import GameCaptcha.src.config as config
from keras.layers import Conv2D, Flatten, Dense, Reshape, Input, Lambda
from keras.models import Model

from GameCaptcha.src.networks_builders.vae import Sampling


def build_encoder(input_shape):
    encoder_input = Input(shape=input_shape, name="encoder_input")

    # Convolutional layers to reduce spatial dimensions
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(encoder_input)  # 128x24
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 64x12
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 32x6
    latent_feature_map = Conv2D(config.latent_channels, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 16x3

    # Parameterize the latent space
    z_mean = Conv2D(config.latent_channels, (1, 1), padding="same", name="z_mean")(latent_feature_map)
    z_log_var = Conv2D(config.latent_channels, (1, 1), padding="same", name="z_log_var")(latent_feature_map)

    z = Sampling()([z_mean, z_log_var])

    # Define the encoder model
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

    return encoder