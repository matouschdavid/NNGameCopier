import matplotlib.pyplot as plt
import numpy as np

from GameCaptcha.src.game_utils import predict_next_frame, update_latent_space_buffer, clean_image, \
    remove_input_from_latent_space


def plot_reconstruction(frames, vae, size=10):
    sample_indices = np.random.choice(len(frames), size=size, replace=False)
    sample_frames = frames[sample_indices]

    reconstructed_frames, z = vae(sample_frames)

    plt.figure(figsize=(size*2, 2))
    for i in range(size):
        plt.subplot(2, size, i + 1)
        plt.imshow(sample_frames[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        plt.subplot(2, size, i + 1 + size)
        plt.imshow(reconstructed_frames[i].numpy().squeeze(), cmap="gray")
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

def plot_prediction(last_frames, inputs_at_start, frames_to_predict, decoder, lstm):
    last_image = clean_image(decoder(remove_input_from_latent_space(np.expand_dims(last_frames[-1], axis=0), len(inputs_at_start[0]))))
    image_index = 1
    plt.figure(figsize=(10, 6))
    for input_at_start in inputs_at_start:
        plt.subplot(len(inputs_at_start), frames_to_predict + 1, image_index)
        plt.imshow(last_image, cmap="gray")
        plt.axis("off")
        plt.title("Start frame")
        image_index += 1

        for i in range(frames_to_predict):
            plt.subplot(len(inputs_at_start), frames_to_predict + 1, image_index)
            if i == 0:
                plt.title(f"{input_at_start}")
                next_image, next_latent_space = predict_next_frame(decoder, lstm, last_frames, input_at_start)
            else:
                next_image, next_latent_space = predict_next_frame(decoder, lstm, last_frames, [0, 0])
            last_frames = update_latent_space_buffer(last_frames, next_latent_space)

            next_image = clean_image(next_image)
            plt.imshow(next_image, cmap="gray")
            plt.axis("off")
            image_index += 1
    plt.show()