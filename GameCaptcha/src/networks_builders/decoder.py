import GameCaptcha.src.config as config
from keras import Model, layers

def build_decoder(latent_height, latent_width, frame_channels):
    latent_input = layers.Input(shape=(latent_height, latent_width, config.latent_channels,), name="latent_input")

    # Transposed convolutional layers to reconstruct spatial dimensions
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(latent_input)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.ReLU()(x)
    decoded = layers.Conv2DTranspose(frame_channels, (3, 3), strides=(2, 2), activation="sigmoid", padding="same")(x)

    decoder = Model(latent_input, decoded, name="decoder")
    return decoder