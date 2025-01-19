import os

os.environ['HSA_OVERRIDE_GFX_VERSION']="11.0.0"

#====================================================================== For Dino

input_keys = ["Key.space", "Key.down"]

compressed_frame_resolution = (256, 48)
frame_channels = "L"
output_frame_resolution = (1024, 256)
downscale_factor = 16
latent_channels = 64
latent_shape = (int(compressed_frame_resolution[1] / downscale_factor), int(compressed_frame_resolution[0] / downscale_factor), latent_channels)
max_time = 987
input_prominence = 1
time_dim = 1
sequence_length = 120
chunk_size = 1000
target_frame_rate = 15

captured_folder = "captured_frames_dino"
compressed_folder = "compressed_frames_dino"
lstm_model_path = "models_dino/lstm.keras"
encoder_model_path = "models_dino/encoder.keras"
decoder_model_path = "models_dino/decoder.keras"

#====================================================================== For Snake

# input_keys = ["Key.up", "Key.down", "Key.left", "Key.right"]
#
# compressed_frame_resolution = (128, 96)
# frame_channels = "RGB"
# output_frame_resolution = (1024, 256)
# downscale_factor = 16
# latent_channels = 64
# latent_shape = (int(compressed_frame_resolution[1] / downscale_factor), int(compressed_frame_resolution[0] / downscale_factor), latent_channels)
# max_time = 514
# input_prominence = 1
# time_dim = 1
# sequence_length = 120
# chunk_size = 1000
# target_frame_rate = 15
#
# captured_folder = "captured_frames_snake"
# compressed_folder = "compressed_frames_snake"
# lstm_model_path = "models_snake/lstm.keras"
# encoder_model_path = "models_snake/encoder.keras"
# decoder_model_path = "models_snake/decoder.keras"
