import os

os.environ['HSA_OVERRIDE_GFX_VERSION']="11.0.0"

compressed_frame_resolution = (256, 48)
downscale_factor = 16
latent_channels = 64
max_time = 5898
input_prominence = 3
time_dim = 1
sequence_length = 120
chunk_size = 1000

captured_folder = "captured_frames"
compressed_folder = "compressed_frames"
lstm_model_path = "models/lstm.keras"
encoder_model_path = "models/encoder.keras"
decoder_model_path = "models/decoder.keras"
