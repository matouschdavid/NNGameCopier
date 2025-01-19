from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.io_utils import load_data, ImageDataGeneratorEager, ImageDataGenerator
from GameCaptcha.src.plot_utils import plot_loss, plot_reconstruction
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from GameCaptcha.src.vae import Sampling, VAE


def train_vae_main(encoder_path, decoder_path, epochs=100, batch_size=32):
    image_folder = "compressed_frames"
    input_file = "compressed_frames/key_logs.txt"

    (input_width, input_height) = NNGCConstants.compressed_image_size
    input_channels = 1 if NNGCConstants.color_mode == 'L' else 3

    latent_dim = NNGCConstants.latent_dimension
    print(f"Latent Dimension: {latent_dim}")

    # Create data generator
    train_generator = ImageDataGenerator(
        image_folder=image_folder,
        input_file=input_file,
        batch_size=batch_size
    )

    encoder_inputs = keras.Input(shape=(input_height, input_width, input_channels))
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(int(input_height / 8) * int(input_width / 8) * input_channels, activation="relu")(latent_inputs)
    x = layers.Reshape((int(input_height / 8), int(input_width / 8), input_channels))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(input_channels, 3, activation="tanh", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    history = vae.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator)
    )

    print("Done training")
    plot_loss(history)

    sample_batch = next(iter(train_generator))[0]
    plot_reconstruction(sample_batch, vae)

    vae.encoder.save(encoder_path)
    vae.decoder.save(decoder_path)

if __name__ == '__main__':
    train_vae_main()