import matplotlib.pyplot as plt
import GameCaptcha.src.config as config
import numpy as np


def plot_reconstruction(frames, encoder, decoder, size=10):
    sample_indices = np.random.choice(len(frames), size=size, replace=False)
    sample_frames = frames[sample_indices]

    latent_spaces = encoder.predict(sample_frames)
    reconstructed_frames = decoder.predict(latent_spaces)

    plt.figure(figsize=(size*2, 2))
    for i in range(size):
        plt.subplot(2, size, i + 1)
        plt.imshow(sample_frames[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        plt.subplot(2, size, i + 1 + size)
        plt.imshow(reconstructed_frames[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Reconstructed")

    plt.show()

def plot_frame(frame):
    plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.imshow(frame.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Frame")

    plt.show()

def plot_sequence(frames):
    plt.figure(figsize=(20, 4))
    for i in range(len(frames)):
        plt.subplot(1, 10, i + 1)
        plt.imshow(frames[i], cmap="gray")
        plt.axis("off")
        plt.title("Frame")

    plt.show()

def plot_loss(history):
    # Extract the loss and validation loss
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    # Check if validation loss is available
    has_val_loss = val_loss is not None

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss', color='blue', marker='o')
    if has_val_loss:
        plt.plot(val_loss, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def predict_sequence(encoder, decoder, lstm, frames, inputs, times, frames_to_predict, input_prominence, input_dim, inputs_at_start):
    """
    Predict a sequence of frames starting from the last `sequence_length` frames, inputs, and times.
    """
    # Step 1: Initialize the latent space buffer
    # Encode the last `sequence_length` frames to latent space
    start_encoder_buffer = np.expand_dims(encoder.predict(frames), axis=0)  # Add batch dimension


    # Add batch and sequence dimensions
    start_input_buffer = np.expand_dims(np.tile(inputs, input_prominence), axis=0)  # Repeat inputs for prominence
    start_time_buffer = np.expand_dims(times, axis=0)  # Add batch dimension

    # Step 3: Predict frames iteratively
    predicted_frames = []

    for input_at_start in inputs_at_start:
        latent_space_buffer = start_encoder_buffer
        input_sequence = start_input_buffer
        time_sequence = start_time_buffer

        predicted_frames.append(decoder.predict(latent_space_buffer[-1])[0])

        for _ in range(frames_to_predict):
            # Predict the next latent space
            next_latent_space = lstm.predict([latent_space_buffer, input_sequence, time_sequence])

            # Decode the next latent space to reconstruct the frame
            next_latent_space_cleaned = next_latent_space[:, :-(config.time_dim + input_dim)]  # Remove time from latent space
            height, width, channels = encoder.output.shape[1:]  # Latent shape from encoder output
            next_latent_space_cleaned = next_latent_space_cleaned.reshape((-1, height, width, channels))

            next_frame = decoder.predict(next_latent_space_cleaned)

            # Store the predicted frame
            predicted_frames.append(next_frame[0])  # Remove batch dimension

            # Update the latent space buffer
            latent_space_buffer = np.roll(latent_space_buffer, shift=-1, axis=1)  # Shift latent space buffer
            latent_space_buffer[0, -1, :] = next_latent_space_cleaned[0]  # Add the new latent space

            # Update the input and time sequences
            input_sequence = np.roll(input_sequence, shift=-1, axis=1)  # Shift inputs
            input_sequence[0, -1, :] = input_at_start  # Assume no new input (can modify as needed)

            last_time_value = time_sequence[0, -1]
            time_sequence = np.roll(time_sequence, shift=-1, axis=1)  # Shift times
            time_sequence[0, -1] = last_time_value + 1/config.max_time  # Assume no new time increment (modify as needed)

    return predicted_frames


def plot_frames(predicted_frames, frames_to_predict):
    """
    Plots a sequence of predicted frames.

    Parameters:
    - predicted_frames: List of predicted frames (images as numpy arrays).
    """
    num_frames = len(predicted_frames)
    plt.figure(figsize=(15, 5))

    input_count = int(num_frames / frames_to_predict)
    print(num_frames, frames_to_predict, input_count)

    for r in range(input_count):
        for c in range(frames_to_predict):
            plt.subplot(input_count, frames_to_predict, r * frames_to_predict + c + 1)

            frame = predicted_frames[r * frames_to_predict + c]
            plt.imshow(frame.squeeze(), cmap="gray")
            plt.axis("off")
            plt.title(f"Frame {c + 1}")

    plt.tight_layout()
    plt.show()