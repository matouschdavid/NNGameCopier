import config
from keras.layers import Conv2D, Flatten, Dense, Reshape, Input
from keras.models import Model

def build_encoder(input_shape):
    encoder_input = Input(shape=input_shape, name="encoder_input")

    # Convolutional layers to reduce spatial dimensions
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(encoder_input)  # 128x24
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 64x12
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 32x6
    latent_feature_map = Conv2D(config.latent_channels, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 16x3

    # Define the encoder model
    encoder = Model(encoder_input, latent_feature_map, name="encoder")

    return encoder