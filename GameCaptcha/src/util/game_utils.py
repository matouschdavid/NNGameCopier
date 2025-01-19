import numpy as np
import GameCaptcha.src.config as config

def predict_next_frame(decoder, lstm_model, encoder_part, input_part, time_part, input_vector):
    """Predicts the next frame given a sequence of latent vectors."""
    next_latent = lstm_model.predict([encoder_part[np.newaxis, ...], input_part[np.newaxis, ...], time_part[np.newaxis, ...]], verbose=0)  # Predict next latent

    latent_only = remove_input_from_latent_space(next_latent, len(input_vector))
    
    latent_only = np.reshape(latent_only, config.latent_shape)[np.newaxis, ...]

    next_frame = decoder(latent_only)

    input_vector = np.tile(input_vector, config.input_prominence)
    input_vector = np.expand_dims(input_vector, axis=0)
    timestamp = time_part[-1][-1] + (1 / config.max_time)
    if timestamp > 0.8:
        print("Reset time")
        timestamp = 0
    timestamp = np.expand_dims(timestamp, axis=0)
    timestamp = np.expand_dims(timestamp, axis=0)

    return next_frame, latent_only, input_vector, timestamp

    
def remove_input_from_latent_space(latent_space, input_dim):
    return latent_space[:, :-(input_dim * config.input_prominence + config.time_dim)]

def clean_image(image):
    image = (image.numpy() * 255).astype(np.uint8)

    image = np.squeeze(image, axis=0)  # Remove batch dimension
    image = np.squeeze(image, axis=-1)  # Remove channel dimension

    return image

def encode_frames(encoder, frames, inputs, timestamps):
    z = encoder(frames)
    z = z.numpy()

    inputs = np.array(inputs)
    inputs = np.tile(inputs, config.input_prominence)

    timestamps = np.expand_dims(timestamps, axis=1)

    return z, inputs, timestamps