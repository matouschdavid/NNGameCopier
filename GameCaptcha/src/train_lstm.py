from tensorflow.keras.models import load_model

from io_utils import load_data
from networks_builders.lstm import build_combined_lstm
from networks_builders.shape_helper import get_shapes
import numpy as np
import random
import os

os.environ['HSA_OVERRIDE_GFX_VERSION']="11.0.0"

def prepare_sequences(encoder, frames, inputs, timestamps, sequence_length):
    latent_sequences = []
    input_sequences = []
    time_sequences = []
    output_sequences = []

    latent_cache = {}

    for i in range(len(frames) - sequence_length):
        frame_seq = frames[i:i + sequence_length]
        input_seq = inputs[i:i + sequence_length]
        time_seq = timestamps[i:i + sequence_length]

        # try getting values from cache
        latent_seq = get_latent_representations(encoder, frame_seq)

        latent_sequences.append(latent_seq)
        input_sequences.append(input_seq)
        time_sequences.append(time_seq)

        predict_frame = frames[i + sequence_length]
        height, width, channels = predict_frame.shape
        predict_frame = predict_frame.reshape(-1, height, width,
                                              channels)

        encoder_part = encoder.predict(predict_frame).flatten()
        encoder_part = np.expand_dims(encoder_part, -1)
        input_part = inputs[i + sequence_length]
        time_part = timestamps[i + sequence_length]

        input_part = np.tile(input_part, 1)  # todo add prominence
        input_part = np.expand_dims(input_part, axis=-1)
        time_part = np.expand_dims(time_part, axis=-1)
        time_part = np.expand_dims(time_part, axis=0)

        output_seq = np.concatenate([encoder_part, input_part, time_part], axis=0)
        output_sequences.append(output_seq)

        # cache all the latent_seq values

    # Convert to numpy arrays
    latent_sequences = np.array(latent_sequences)
    input_sequences = np.array(input_sequences)
    time_sequences = np.array(time_sequences)
    output_sequences = np.array(output_sequences)

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

encoder = load_model("models/encoder.keras")
decoder = load_model("models/decoder.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file) # load every frame, input and timestamp
input_shape, latent_shape = get_shapes(frames)
input_dim = inputs.shape[-1]
input_prominence = 3
time_dim = 1
sequence_length = 120

lstm_model = build_combined_lstm(latent_shape, input_dim, input_prominence, time_dim, sequence_length)

chunks = []
chunk_size = 200
for k in range(1):
    for i in range(0, len(frames), chunk_size):
        if i + chunk_size > len(frames):
            rest_size = len(frames) - i
            if rest_size < sequence_length:
                chunks.remove((i-chunk_size,i))
                chunks.append((i-chunk_size, i + rest_size))
            else:
                chunks.append((i, i + rest_size))
        else:
            chunks.append((i,i + chunk_size))

random.shuffle(chunks)

counter = 0
for fr, to in chunks:
    # Prepare the sequences
    frames_chunk = frames[fr:to]
    inputs_chunk = inputs[fr:to]
    timestamps_chunk = timestamps[fr:to]
    print(f"({counter} / {len(chunks)})")
    latent_sequences, input_sequences, time_sequences, output_sequences = prepare_sequences(encoder, frames_chunk, inputs_chunk, timestamps_chunk, sequence_length)
    print(latent_sequences.shape, input_sequences.shape, time_sequences.shape, output_sequences.shape)
    # Train the LSTM
    history = lstm_model.fit(
        [latent_sequences, input_sequences, time_sequences],
        output_sequences,
        batch_size=32,
        epochs=50,
        validation_split=0.2
    )
    counter += 1

lstm_model.save("models/lstm.keras")

# jump = [1, 0]
# duck = [0, 1]
# nothing = [0, 0]
# inputs_at_start = [jump, duck, nothing]
# frames_to_predict = 5
#
# initial_frames = frames[:sequence_length]
# input_vectors = inputs[:sequence_length]
# time_values = timestamps[:sequence_length]
#
# predicted_frames = predict_sequence(
#     encoder, decoder, lstm_model, initial_frames, input_vectors, time_values, frames_to_predict, 1, input_dim, time_dim
# ) # todo add prominence
#
# # Visualize the predicted frames
# plot_frames(predicted_frames)


# # Unfreeze the encoder and decoder layers for fine-tuning
# encoder.trainable = True
# decoder.trainable = True
#
# # Re-compile the model after unfreezing
# lstm_model.compile(optimizer=Adam(learning_rate=1e-5), loss="mse")
#
# # Fine-tune the model
# history_finetune = lstm_model.fit(
#     [latent_sequences, input_sequences, time_sequences],  # Inputs
#     latent_sequences,  # Target (predicting the latent space)
#     batch_size=32,
#     epochs=50,
#     validation_split=0.2
# )