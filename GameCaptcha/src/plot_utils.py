import matplotlib.pyplot as plt
import numpy as np

from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.train_frame_predictor_v3 import predict_next_frame


def plot_reconstruction(frames, vae, size=10):
    sample_indices = np.random.choice(len(frames), size=size, replace=False)
    sample_frames = frames[sample_indices]

    reconstructed_frames, z = vae(sample_frames)

    plt.figure(figsize=(size*2, 2))
    for i in range(size):
        plt.subplot(2, size, i + 1)
        # plt.imshow(sample_frames[i])
        plt.imshow(np.squeeze(sample_frames[i], axis=-1))
        plt.axis("off")
        plt.title("Original")

        plt.subplot(2, size, i + 1 + size)
        # plt.imshow(reconstructed_frames[i].numpy())
        plt.imshow(reconstructed_frames[i].numpy())
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


# def plot_generated_sequence_flotschi(lstm_model, encoder, decoder, frames, num_frames=5):
#     # Get a random initial frame from your dataset
#     random_idx = np.random.randint(0, len(frames))
#     initial_frame = frames[random_idx:random_idx+1]
#
#     # Encode initial frame
#     _, _, initial_encoded = encoder(initial_frame)
#     current_encoded = initial_encoded.numpy()
#
#     # Create a list to store frames
#     generated_frames = [initial_frame[0]]  # Start with the initial frame
#
#     # Generate subsequent frames
#     for i in range(num_frames):
#         # Generate random action (0 or 1) for demonstration
#         action = np.array([[np.random.randint(0, 1)]])
#
#         # Predict next frame
#         frame_input = current_encoded.reshape(1, 1, -1)
#         action_input = action.reshape(1, 1, -1)
#         next_encoded = predict_next_frame(lstm_model, frame_input, action_input)
#
#         # Decode the frame
#         next_frame = decoder(next_encoded.reshape(1, -1))
#
#         # Add to list of frames
#         generated_frames.append(next_frame[0])
#
#         # Update current encoded frame
#         current_encoded = next_encoded
#
#     # Plot the frames
#     fig, axes = plt.subplots(1, num_frames + 1, figsize=(15, 3))
#     fig.suptitle('Generated Frame Sequence (Initial + 5 Predictions)')
#
#     for i, frame in enumerate(generated_frames):
#         axes[i].imshow(np.squeeze(frame))
#         axes[i].axis('off')
#         if i == 0:
#             axes[i].set_title('Initial')
#         else:
#             axes[i].set_title(f'Pred {i}')
#
#     plt.tight_layout()
#     plt.show()

def plot_generated_sequence_flotschi_seqency(lstm_model, encoder, decoder, frames, inputs, num_frames):
    # Get a random starting point that allows for a full sequence
    random_idx = np.random.randint(0, len(frames) - NNGCConstants.sequence_length - num_frames)

    # Get initial sequence of frames
    initial_sequence = frames[random_idx:random_idx + NNGCConstants.sequence_length]

    # Get the corresponding sequence of inputs
    initial_actions = inputs[random_idx:random_idx + NNGCConstants.sequence_length]
    future_actions = inputs[random_idx + NNGCConstants.sequence_length:
                          random_idx + NNGCConstants.sequence_length + num_frames]

    # Encode the sequence
    _, _, encoded_sequence = encoder(initial_sequence)
    current_encoded_sequence = encoded_sequence.numpy()

    # Create a list to store frames
    generated_frames = [initial_sequence[-1]]  # Start with the last frame of initial sequence

    # Create sequence of actions from real inputs
    action_sequence = np.array([action for action in initial_actions])

    # Generate subsequent frames
    for i in range(num_frames):
        # Get the next real action
        new_action = np.array([future_actions[i]])

        # Predict next frame using the sequence
        next_encoded = predict_next_frame(
            lstm_model,
            current_encoded_sequence,
            action_sequence
        )

        # Decode the frame
        next_frame = decoder(next_encoded.reshape(1, -1))

        # Add to list of frames
        generated_frames.append(next_frame[0])

        # Update sequences for next iteration
        # Remove oldest frame and add new frame to encoded sequence
        current_encoded_sequence = np.roll(current_encoded_sequence, -1, axis=0)
        current_encoded_sequence[-1] = next_encoded

        # Update action sequence
        action_sequence = np.roll(action_sequence, -1, axis=0)
        action_sequence[-1] = new_action

    # Plot the frames
    fig, axes = plt.subplots(1, num_frames + 1, figsize=(15, 6))
    fig.suptitle(f'Generated Frame Sequence (Initial + {num_frames} Predictions)')

    actions_to_plot = np.concatenate([initial_actions[-1:], future_actions[:num_frames]])

    # Plot frames
    for i, frame in enumerate(generated_frames):
        axes[i].imshow(np.squeeze(frame))
        axes[i].axis('off')

        if i == 0:
            title = 'Initial'
        else:
            title = f'Pred {i}'
        axes[i].set_title(title + f"\n{actions_to_plot[i]}")

    plt.tight_layout()
    plt.show()