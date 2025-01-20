import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.plot_utils import plot_generated_sequence
from GameCaptcha.src.train_frame_predictor import PositionalEncoding
from GameCaptcha.src.vae import Sampling

postfix = NNGCConstants.postfix

encoder_path = f"{NNGCConstants.model_path}vae_encoder{postfix}.keras"
decoder_path = f"{NNGCConstants.model_path}vae_decoder{postfix}.keras"
predictor_path = f"{NNGCConstants.model_path}lstm_model{postfix}.keras"

encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
decoder = load_model(decoder_path)
lstm_model = load_model(predictor_path, custom_objects={'PositionalEncoding': PositionalEncoding}, safe_mode=False)

image_folder = NNGCConstants.image_path
input_file = NNGCConstants.input_file
frames, inputs, _ = load_data(image_folder, input_file, max=1000)

print("Generating multiple sequences...")
for i in range(10):
    print(f"\nSequence {i + 1}:")
    plot_generated_sequence(lstm_model, encoder, decoder, frames, inputs, 5)
