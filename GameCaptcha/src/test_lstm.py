from tensorflow.keras.models import load_model

from io_utils import load_data
from networks_builders.shape_helper import get_shapes

from plot_utils import predict_sequence, plot_frames

encoder = load_model("models/encoder.keras")
decoder = load_model("models/decoder.keras")
lstm_model = load_model("models/lstm.keras")

image_folder = "compressed_frames"
input_file = "compressed_frames/key_logs.txt"
frames, inputs, timestamps = load_data(image_folder, input_file, max=200)
input_shape, latent_shape = get_shapes(frames)
input_dim = inputs.shape[-1]
input_prominence = 3
time_dim = 1
sequence_length = 120

up = [1, 0, 0, 0]
down = [0, 1, 0, 0]
left = [0, 0, 1, 0]
right = [0, 0, 0, 1]
nothing = [0, 0, 0, 0]
inputs_at_start = [up, down, left, right, nothing]
frames_to_predict = 5

initial_frames = frames[:sequence_length]
input_vectors = inputs[:sequence_length]
time_values = timestamps[:sequence_length]

predicted_frames = predict_sequence(
    encoder, decoder, lstm_model, initial_frames, input_vectors, time_values, frames_to_predict, 1, input_dim, time_dim, inputs_at_start
) # todo add prominence

# Visualize the predicted frames
plot_frames(predicted_frames)