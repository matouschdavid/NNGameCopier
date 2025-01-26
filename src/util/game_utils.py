import numpy as np
import src.config as config

def predict_next_frame(decoder, lstm, latent_space_buffer, input_sequence, time_sequence, new_input):
    # Predict the next latent space
    batched_buffer = np.expand_dims(latent_space_buffer, axis=0)
    next_latent_space = lstm.predict([batched_buffer, input_sequence, time_sequence])

    # Decode the next latent space to reconstruct the frame
    next_latent_space_cleaned = next_latent_space[:, :-(config.time_dim + len(new_input))]  # Remove time from latent space
    height, width, channels = config.latent_shape  # Latent shape from encoder output
    next_latent_space_cleaned = next_latent_space_cleaned.reshape((-1, height, width, channels))

    next_frame = decoder.predict(next_latent_space_cleaned)

    # Update the latent space buffer
    latent_space_buffer = np.roll(latent_space_buffer, shift=-1, axis=0)  # Shift latent space buffer
    latent_space_buffer[-1, :] = next_latent_space_cleaned[0]  # Add the new latent space

    # Update the input and time sequences
    input_sequence = np.roll(input_sequence, shift=-1, axis=1)  # Shift inputs
    input_sequence[0, -1, :] = new_input  # Assume no new input (can modify as needed)

    last_time_value = time_sequence[0, -1]
    time_sequence = np.roll(time_sequence, shift=-1, axis=1)  # Shift times
    time_sequence[0, -1] = last_time_value + 1/config.max_time  # Assume no new time increment (modify as needed)

    return next_frame, latent_space_buffer, input_sequence, time_sequence

    
def remove_input_from_latent_space(latent_space, input_dim):
    return latent_space[:, :-(input_dim * config.input_prominence + config.time_dim)]

def clean_image(image):
    image = (image * 255).astype(np.uint8)

    image = np.squeeze(image, axis=0)  # Remove batch dimension
    if len(config.frame_channels) == 1:
        image = np.squeeze(image, axis=-1)  # Remove channel dimension

    return image

def encode_frames(encoder, frames, inputs, timestamps):
    z = encoder(frames)
    z = z.numpy()

    inputs = np.array(inputs)
    inputs = np.tile(inputs, config.input_prominence)

    timestamps = np.expand_dims(timestamps, axis=1)

    return z, inputs, timestamps