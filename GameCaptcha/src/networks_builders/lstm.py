import GameCaptcha.src.config as config
from keras import layers, Model
import numpy as np

def build_combined_lstm(latent_shape, input_dim):
    # Latent shape from encoder
    height, width, channels = latent_shape

    # Inputs
    encoder_input = layers.Input(shape=(config.sequence_length, height, width, channels), name="encoder_input")
    input_vector = layers.Input(shape=(config.sequence_length, input_dim), name="input_vector")
    time_input = layers.Input(shape=(config.sequence_length, config.time_dim), name="time_input")

    # ConvLSTM for encoder output
    conv_lstm_output = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding="same", return_sequences=True, name="conv_lstm"
    )(encoder_input)

    # Flatten the ConvLSTM output while keeping the sequence dimension
    flat_conv_lstm_output = layers.TimeDistributed(layers.Flatten(), name="flatten_conv_lstm")(conv_lstm_output)

    # Repeat the flattened input vector across sequence length
    # repeated_input_vector = layers.RepeatVector(input_prominence, name="Repeat Input")(input_vector)

    # Combine ConvLSTM output with the repeated input vector
    combined_main_input = layers.Concatenate(name="combine_inputs")([flat_conv_lstm_output, input_vector, time_input])

    # LSTM for combined input
    x = layers.LSTM(128, return_sequences=True)(combined_main_input)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # Final output layer
    latent_dim = height * width * channels  # Dimension of the latent representation from the encoder
    total_input_dim = latent_dim + input_dim  # Combining latent representation and input vector
    output_dim = total_input_dim + config.time_dim  # Final output dimension

    # Output layer for the LSTM model
    lstm_outputs = layers.Dense(output_dim, name="output_dense")(x)

    # Define and compile the model
    lstm_model = Model(inputs=[encoder_input, input_vector, time_input], outputs=lstm_outputs, name="combined_lstm_model")
    lstm_model.compile(optimizer="adam", loss="mse")

    return lstm_model

def prepare_sequences(encoder, frames, inputs, timestamps):
    latent_sequences = []
    input_sequences = []
    time_sequences = []
    output_sequences = []

    latent_cache = {}  # Initialize the cache for individual frames

    def get_latent_cached(frame):
        """Retrieve or compute the latent representation for a single frame."""
        frame_key = hash(frame.tobytes())
        if frame_key not in latent_cache:
            # Expand dimensions of the frame to simulate a batch with a single frame
            frame_expanded = np.expand_dims(frame, axis=0)  # Shape: (1, height, width, channels)
            # Compute the latent representation and cache it
            latent_cache[frame_key] = encoder.predict(frame_expanded)[0]  # Remove batch dimension after prediction
        return latent_cache[frame_key]  # Shape: (latent_height, latent_width, latent_channels)

    for i in range(len(frames) - config.sequence_length):
        frame_seq = frames[i:i + config.sequence_length]
        input_seq = inputs[i:i + config.sequence_length]
        time_seq = timestamps[i:i + config.sequence_length]

        # Retrieve latent representations for the sliding window
        latent_seq = np.stack([get_latent_cached(frame) for frame in frame_seq])  # Shape: (sequence_length, latent_height, latent_width, latent_channels)
        #print(latent_seq.shape)

        latent_sequences.append(latent_seq)
        input_sequences.append(input_seq)
        time_sequences.append(time_seq)

        predict_frame = frames[i + config.sequence_length]
        height, width, channels = predict_frame.shape
        predict_frame = predict_frame.reshape(-1, height, width, channels)

        encoder_part = encoder.predict(predict_frame).flatten()
        encoder_part = np.expand_dims(encoder_part, -1)
        input_part = inputs[i + config.sequence_length]
        time_part = timestamps[i + config.sequence_length]

        input_part = np.tile(input_part, 1)  # todo add prominence
        input_part = np.expand_dims(input_part, axis=-1)
        time_part = np.expand_dims(time_part, axis=-1)
        time_part = np.expand_dims(time_part, axis=0)

        output_seq = np.concatenate([encoder_part, input_part, time_part], axis=0)
        output_sequences.append(output_seq)

    # Convert to numpy arrays
    latent_sequences = np.array(latent_sequences)  # Shape: (batch_size, sequence_length, height, width, channels)
    input_sequences = np.array(input_sequences)  # Shape: (batch_size, sequence_length, input_dim)
    time_sequences = np.array(time_sequences)  # Shape: (batch_size, sequence_length, time_dim)
    output_sequences = np.array(output_sequences)  # Shape: (batch_size, output_dim)

    print("Shape of latent sequences: ", latent_sequences.shape)
    print("Shape of input sequences: ", input_sequences.shape)
    print("Shape of time sequences: ", time_sequences.shape)
    print("Shape of output sequences: ", output_sequences.shape)

    return latent_sequences, input_sequences, time_sequences, output_sequences

def get_latent_representation(encoder, frame):
    height, width, channels = frame.shape
    frame = frame.reshape(-1, height, width, channels)

    return encoder.predict(frame)

def get_latent_representations(encoder, frame_sequences):
    """
    Get latent representations for a sequence of frames.

    Parameters:
    - encoder: Trained encoder model.
    - frame_sequences: Numpy array of frame sequences (batch_size, sequence_length, height, width, channels).

    Returns:
    - latent_sequences: Latent representations (batch_size, sequence_length, latent_height, latent_width, latent_channels).
    """
    # Flatten sequence dimension into the batch dimension
    sequence_length, height, width, channels = frame_sequences.shape
    reshaped_frames = frame_sequences.reshape(-1, height, width,
                                              channels)  # Shape: (batch_size * sequence_length, height, width, channels)

    # Get latent space representations for each frame
    latent_frames = encoder.predict(
        reshaped_frames)  # Shape: (batch_size * sequence_length, latent_height, latent_width, latent_channels)

    # Reshape back to sequence format
    latent_sequences = latent_frames.reshape(sequence_length, *latent_frames.shape[
                                                                           1:])  # Shape: (batch_size, sequence_length, latent_height, latent_width, latent_channels)

    return latent_sequences
