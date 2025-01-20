import GameCaptcha.src.config as config
from tensorflow.keras.models import load_model
import random

from GameCaptcha.src.networks_builders.lstm import build_combined_lstm, prepare_sequences
from GameCaptcha.src.networks_builders.vae import Sampling
from GameCaptcha.src.util.io_utils import load_data

encoder = load_model(config.encoder_model_path, custom_objects={"Sampling": Sampling})
decoder = load_model(config.decoder_model_path)

frames, inputs, timestamps = load_data(config.compressed_folder, max=2000) # load every frame, input and timestamp
max_time = max(timestamps)
print("Max time of dataset", max_time)
timestamps = timestamps / max_time
input_dim = inputs.shape[-1]
chunks = []

lstm_model = build_combined_lstm(config.latent_shape, input_dim)

for k in range(1):
    for i in range(0, len(frames), config.chunk_size):
        if i + config.chunk_size > len(frames):
            rest_size = len(frames) - i
            if rest_size < config.sequence_length:
                chunks.remove((i-config.chunk_size,i))
                chunks.append((i-config.chunk_size, i + rest_size))
            else:
                chunks.append((i, i + rest_size))
        else:
            chunks.append((i,i + config.chunk_size))

random.shuffle(chunks)

counter = 0
for fr, to in chunks:
    # Prepare the sequences
    frames_chunk = frames[fr:to]
    inputs_chunk = inputs[fr:to]
    timestamps_chunk = timestamps[fr:to]
    print(f"({counter} / {len(chunks)})")
    latent_sequences, input_sequences, time_sequences, output_sequences = prepare_sequences(encoder, frames_chunk, inputs_chunk, timestamps_chunk)
    # Train the LSTM
    history = lstm_model.fit(
        [latent_sequences, input_sequences, time_sequences],
        output_sequences,
        batch_size=32,
        epochs=30,
        validation_split=0.2
    )
    counter += 1
print("Done training")

lstm_model.save(config.lstm_model_path)
print("Model saved")
print("Max time of dataset", max_time)
