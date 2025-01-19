import numpy as np

def predict_next_frame(decoder, lstm_model, encoder_part, input_part, time_part, max_time, input_vector, input_prominence, time_dim, latent_shape, bot=False):
    """Predicts the next frame given a sequence of latent vectors."""
    next_latent = lstm_model.predict([encoder_part[np.newaxis, ...], input_part[np.newaxis, ...], time_part[np.newaxis, ...]], verbose=0)  # Predict next latent

    latent_only = remove_input_from_latent_space(next_latent, len(input_vector), input_prominence, time_dim)
    
    latent_only = np.reshape(latent_only, latent_shape)[np.newaxis, ...]

    next_frame = decoder(latent_only)
    
    if not bot:
        input_vector = np.tile(input_vector, input_prominence)
        input_vector = np.expand_dims(input_vector, axis=0)
        timestamp = time_part[-1][-1] + (1 / max_time)
        if timestamp > 0.8:
            print("Reset time")
            timestamp = 0
        timestamp = np.expand_dims(timestamp, axis=0)
        timestamp = np.expand_dims(timestamp, axis=0)

    return next_frame, latent_only, input_vector, timestamp

    
def remove_input_from_latent_space(latent_space, input_dim, input_prominence, time_dim):
    return latent_space[:, :-(input_dim * input_prominence + time_dim)]

def update_latent_space_buffer(latent_space_buffer, new_latent_space):
    # Roll the buffer to the left (remove the first element)
    latent_space_buffer = np.roll(latent_space_buffer, shift=-1, axis=0)
    # Add the new latent space at the end of the buffer
    latent_space_buffer[-1] = new_latent_space
    return latent_space_buffer

def clean_image(image):
    image = (image.numpy() * 255).astype(np.uint8)

    image = np.squeeze(image, axis=0)  # Remove batch dimension
    image = np.squeeze(image, axis=-1)  # Remove channel dimension

    return image

def encode_frames(encoder, frames, inputs, timestamps, input_prominence):
    z = encoder(frames)
    z = z.numpy()

    inputs = np.array(inputs)
    inputs = np.tile(inputs, input_prominence)

    timestamps = np.expand_dims(timestamps, axis=1)

    return z, inputs, timestamps