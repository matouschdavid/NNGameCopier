import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input, LSTM, Dropout, BatchNormalization
import matplotlib.pyplot as plt

from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.io_utils import load_data, LSTMImageDataGeneratorEager
from tensorflow.keras.models import load_model


image_folder = NNGCConstants.image_path
input_file = NNGCConstants.input_file

def create_lstm_model(latent_dim=128, num_actions=1):
    # Input layers
    latent_input = layers.Input(shape=(NNGCConstants.sequence_length, latent_dim))  # Changed from (1, latent_dim)
    action_input = layers.Input(shape=(NNGCConstants.sequence_length, num_actions*NNGCConstants.action_weight))  # Changed from (1, num_actions)

    combined_input = layers.Concatenate(axis=-1)([latent_input, action_input])
    lstm1 = layers.LSTM(256, return_sequences=True)(combined_input)
    lstm2 = layers.LSTM(256)(lstm1)
    dense1 = layers.Dense(512, activation='relu')(lstm2)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    output = layers.Dense(latent_dim)(dense2)

    model = keras.Model(inputs=[latent_input, action_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru_model(latent_dim=128, num_actions=1):
    latent_input = layers.Input(shape=(NNGCConstants.sequence_length, latent_dim))
    action_input = layers.Input(shape=(NNGCConstants.sequence_length, num_actions*NNGCConstants.action_weight))

    combined_input = layers.Concatenate(axis=-1)([latent_input, action_input])
    gru1 = layers.GRU(256, return_sequences=True)(combined_input)
    gru2 = layers.GRU(256)(gru1)
    dense1 = layers.Dense(512, activation='relu')(gru2)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    output = layers.Dense(latent_dim)(dense2)

    model = keras.Model(inputs=[latent_input, action_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Create the positional encoding matrix during initialization
        self.pos_encoding = self._positional_encoding()

    def _get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return pos * angle_rates

    def _positional_encoding(self):
        angle_rads = self._get_angles(
            np.arange(self.sequence_length)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config

def create_transformer_model(latent_dim=128, num_actions=1):
    latent_input = layers.Input(shape=(NNGCConstants.sequence_length, latent_dim))
    action_input = layers.Input(shape=(NNGCConstants.sequence_length, num_actions*NNGCConstants.action_weight))

    combined_input = layers.Concatenate(axis=-1)([latent_input, action_input])

    # Calculate total dimension after concatenation
    total_dim = latent_dim + num_actions*NNGCConstants.action_weight

    # Add positional encoding
    pos_encoding = PositionalEncoding(NNGCConstants.sequence_length, total_dim)
    encoded_input = pos_encoding(combined_input)

    # Transformer layers
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(encoded_input, encoded_input)
    x = layers.LayerNormalization()(x + encoded_input)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(latent_dim)(x)

    model = keras.Model(inputs=[latent_input, action_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_bilstm_attention_model(latent_dim=128, num_actions=1):
    latent_input = layers.Input(shape=(NNGCConstants.sequence_length, latent_dim))
    action_input = layers.Input(shape=(NNGCConstants.sequence_length, num_actions*NNGCConstants.action_weight))

    combined_input = layers.Concatenate(axis=-1)([latent_input, action_input])

    # Bidirectional LSTM layers
    bilstm1 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(combined_input)
    bilstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(bilstm1)

    # Attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(bilstm2, bilstm2)
    attention = layers.LayerNormalization()(attention + bilstm2)

    # Global pooling
    pooled = layers.GlobalAveragePooling1D()(attention)

    dense1 = layers.Dense(512, activation='relu')(pooled)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    output = layers.Dense(latent_dim)(dense2)

    model = keras.Model(inputs=[latent_input, action_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to prepare data for training
def prepare_training_data(encoded_frames, actions):
    X_frames = []
    X_actions = []
    y = []

    for i in range(len(encoded_frames) - NNGCConstants.sequence_length):
        X_frames.append(encoded_frames[i:i+NNGCConstants.sequence_length])
        X_actions_unweighted = actions[i:i+NNGCConstants.sequence_length]
        X_actions.append([action * NNGCConstants.action_weight for action in X_actions_unweighted])
        y.append(encoded_frames[i+NNGCConstants.sequence_length])

    return [np.array(X_frames), np.array(X_actions)], np.array(y)

# Example training code (you'll need to adapt this to your specific data)
def train_model(model, encoded_frames, actions, epochs=100, batch_size=32):
    # Prepare training data
    X, y = prepare_training_data(encoded_frames, actions)

    # Train the model
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )

    return history

# Function to predict the next frame
def predict_next_frame(model, current_encoded_frames, current_actions):
    # Ensure we have the correct sequence length
    if len(current_encoded_frames) < NNGCConstants.sequence_length:
        raise ValueError("Not enough frames for sequence")

    # Take the last sequence_length frames
    frame_sequence = current_encoded_frames[-NNGCConstants.sequence_length:]
    action_sequence = current_actions[-NNGCConstants.sequence_length:]

    # Adjust actions weight
    weighted_actions = [list(actions) * NNGCConstants.action_weight for actions in action_sequence]

    # Reshape inputs for prediction
    frame_input = np.array([frame_sequence])  # Shape: (1, sequence_length, latent_dim)
    action_input = np.array([weighted_actions])  # Shape: (1, sequence_length, num_actions * NNGCConstants.action_weight)

    # Predict next latent state
    next_encoded_frame = model.predict([frame_input, action_input])

    return next_encoded_frame[0]

def train_prediction_main(encoder_path, decoder_path, predictor_path, epochs=100, batch_size=32):
    from GameCaptcha.src.vae import Sampling
    encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
    decoder = load_model(decoder_path)

    # Create the model
    model = create_bilstm_attention_model(
        latent_dim=NNGCConstants.latent_dimension,
        num_actions=NNGCConstants.action_count
    )

    train_generator = LSTMImageDataGeneratorEager(
        image_folder=image_folder,
        input_file=input_file,
        batch_size=batch_size,
        sequence_length=NNGCConstants.sequence_length,
        encoder=encoder
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        batch_size=batch_size
    )

    model.save(predictor_path)

    from GameCaptcha.src.plot_utils import plot_loss

    plot_loss(history)

    from GameCaptcha.src.plot_utils import plot_generated_sequence
    # Plot multiple sequences to evaluate model performance
    print("Generating multiple sequences...")
    sample_frames, sample_inputs, _ = load_data(image_folder, input_file, max=1500)
    for i in range(3):
        print(f"\nSequence {i+1}:")
        plot_generated_sequence(model, encoder, decoder, sample_frames, sample_inputs, 5)

    train_generator.clear_cache()


if __name__ == '__main__':
    train_prediction_main()