import GameCaptcha.src.config as config
from keras import Model
from keras.src.losses import BinaryCrossentropy
import tensorflow as tf

from GameCaptcha.src.networks_builders.decoder import build_decoder
from GameCaptcha.src.networks_builders.encoder import build_encoder


def custom_loss(y_true, y_pred):
    # Basic MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

    # Optional: Add a small term to prevent vanishing gradients
    regularization_term = 1e-5 * tf.reduce_mean(tf.square(y_pred))

    return mse_loss + regularization_term

def build_autoencoder(input_shape, latent_height, latent_width):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(latent_height, latent_width, input_shape[-1])

    autoencoder = Model(encoder.input, decoder(encoder.output))
    autoencoder.compile(optimizer="adam", loss=BinaryCrossentropy(from_logits=False))

    return autoencoder, encoder, decoder
