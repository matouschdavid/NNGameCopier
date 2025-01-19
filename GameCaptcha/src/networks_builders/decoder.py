import GameCaptcha.src.config as config
from keras import Model, layers

def build_decoder(latent_height, latent_width):
    inputs = layers.Input(shape=(latent_height, latent_width, config.latent_channels))
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.ReLU()(x)
    decoded = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), activation="sigmoid", padding="same")(x)  # Output layer with sigmoid
    return Model(inputs, decoded)