from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.plot_utils import plot_loss, plot_reconstruction
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from GameCaptcha.src.vae import Sampling, VAE


def train_vae_main(encoder_path, decoder_path, epochs=100, batch_size=32):
    image_folder = "compressed_frames"
    input_file = "compressed_frames/key_logs.txt"
    frames, _, _ = load_data(image_folder, input_file)

    input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]

    # latent_dim = int(0.05 * input_height * input_width * input_channels)
    latent_dim = NNGCConstants.latent_dimension
    print(f"Latent Dimension: {latent_dim}")

    encoder_inputs = keras.Input(shape=(input_height, input_width, input_channels))
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(int(latent_dim * 1.5), activation="relu")(x)
    # x = layers.Dense(latent_dim, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    # x = layers.Dense(int(latent_dim * 1.5), activation="relu")(latent_inputs)
    x = layers.Dense(int(input_height / 8) * int(input_width / 8) * input_channels, activation="relu")(latent_inputs)
    x = layers.Reshape((int(input_height / 8), int(input_width / 8), input_channels))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="tanh", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    history = vae.fit(frames, epochs=epochs, batch_size=batch_size)
    print("Done training")
    plot_loss(history)
    plot_reconstruction(frames, vae)

    vae.encoder.save(encoder_path)
    vae.decoder.save(decoder_path)

if __name__ == '__main__':
    train_vae_main()