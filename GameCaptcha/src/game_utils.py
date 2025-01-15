import numpy as np

def predict_next_frame(decoder, lstm_model, latent_sequence, input_vector, bot=False):
    """Predicts the next frame given a sequence of latent vectors."""
    next_latent = lstm_model.predict(latent_sequence[np.newaxis, ...])  # Predict next latent

    latent_only = remove_input_from_latent_space(next_latent, len(input_vector))

    if not bot:
        input_vector = np.expand_dims(input_vector, axis=0)
        timestamp = latent_sequence[-1][-1] + 1
        print(timestamp)
        timestamp = np.expand_dims(timestamp, axis=0)
        timestamp = np.expand_dims(timestamp, axis=0)
        next_latent = np.concatenate([latent_only, input_vector, timestamp], axis=-1)
        print(next_latent)

    next_frame = decoder(latent_only)
    return next_frame, next_latent

def remove_input_from_latent_space(latent_space, input_dim):
    return latent_space[:, :-(input_dim)]  # Slice to exclude the appended input dimensions

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

def encode_frames(encoder, frames, inputs, timestamps):
    _, _, z = encoder(frames)
    z = z.numpy()

    inputs = np.array(inputs)

    timestamps = np.expand_dims(timestamps, axis=1)

    combined_z = np.array([np.concatenate([a1, a2, a3]) for a1, a2, a3 in zip(z, inputs, timestamps)])
    return combined_z