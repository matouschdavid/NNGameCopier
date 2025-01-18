import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.plot_utils import plot_generated_sequence_flotschi_seqency
from GameCaptcha.src.train_frame_predictor_v3 import PositionalEncoding
from GameCaptcha.src.vae import Sampling



postfix = "_flappy_128"

encoder_path = f"models/vae_encoder{postfix}.keras"
decoder_path = f"models/vae_decoder{postfix}.keras"
predictor_path = f"models/lstm_model{postfix}.keras"

encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
decoder = load_model(decoder_path)
lstm_model = load_model(predictor_path, custom_objects={'PositionalEncoding': PositionalEncoding}, safe_mode=False)

# Load the data
image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, _ = load_data(image_folder, input_file)

# Generate and plot multiple sequences
print("Generating multiple sequences...")
for i in range(3):
    print(f"\nSequence {i+1}:")
    plot_generated_sequence_flotschi_seqency(lstm_model, encoder, decoder, frames, inputs, 5)