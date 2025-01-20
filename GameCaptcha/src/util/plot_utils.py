import matplotlib.pyplot as plt
import GameCaptcha.src.config as config
import numpy as np

from GameCaptcha.src.util.game_utils import predict_next_frame


def plot_reconstruction(frames, encoder, decoder, size=10):
    sample_indices = np.random.choice(len(frames), size=size, replace=False)
    sample_frames = frames[sample_indices]

    _, _, latent_spaces = encoder.predict(sample_frames)
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
    _, _, start_encoder_buffer = encoder.predict(frames)


    # Add batch and sequence dimensions
    start_input_buffer = np.expand_dims(np.tile(inputs, input_prominence), axis=0)  # Repeat inputs for prominence
    start_time_buffer = np.expand_dims(times, axis=0)  # Add batch dimension

    # Step 3: Predict frames iteratively
    predicted_frames = []
    plot_buffer = start_encoder_buffer

    for input_at_start in inputs_at_start:
        latent_space_buffer = start_encoder_buffer
        input_sequence = start_input_buffer
        time_sequence = start_time_buffer

        predicted_frames.append(decoder.predict(np.expand_dims(latent_space_buffer[-1], axis=0))[0])
        for i in range(frames_to_predict):
            next_frame, latent_space_buffer, input_sequence, time_sequence = predict_next_frame(decoder, lstm, latent_space_buffer, input_sequence, time_sequence, input_at_start)
            predicted_frames.append(next_frame)
        plot_buffer = latent_space_buffer

    reconstructed_frames = decoder.predict(plot_buffer)

    plt.figure(figsize=(config.sequence_length, 1))
    for i in range(config.sequence_length):
        plt.subplot(1, config.sequence_length, i + 1)
        plt.imshow(reconstructed_frames[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(f"{i}")

    plt.show()

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