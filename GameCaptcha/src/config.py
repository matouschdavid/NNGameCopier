import os

os.environ['HSA_OVERRIDE_GFX_VERSION']="11.0.0"

input_keys = ["Key.space", "Key.down"]

compressed_frame_resolution = (256, 48)
output_frame_resolution = (1024, 256)
downscale_factor = 16
latent_channels = 64
latent_shape = (int(compressed_frame_resolution[1] / downscale_factor), int(compressed_frame_resolution[0] / downscale_factor), latent_channels)
max_time = 2439
input_prominence = 1
time_dim = 1
sequence_length = 120
chunk_size = 1000
target_frame_rate = 15

captured_folder = "captured_frames"
compressed_folder = "compressed_frames"
lstm_model_path = "models_dino/lstm.keras"
encoder_model_path = "models_dino/encoder.keras"
decoder_model_path = "models_dino/decoder.keras"
