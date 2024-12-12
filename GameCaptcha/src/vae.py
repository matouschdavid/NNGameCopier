import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import random_normal
from PIL import Image
import matplotlib.pyplot as plt

# VAE Encoder
# VAE Encoder
class VAEEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAEEncoder, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')  # 64 filters
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')  # 128 filters
        self.conv3 = layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')  # 256 filters
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')  # Increased dense layer
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


# VAE Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = random_normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VAEDecoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAEDecoder, self).__init__()
        self.fc = layers.Dense(32 * 8 * 256, activation='relu')  # Match encoder bottleneck output
        self.reshape = layers.Reshape((8, 32, 256))
        self.deconv1 = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')  # Output: 16x16
        self.deconv2 = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')   # Output: 32x32
        self.deconv3 = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')  # Output: 64x64
        self.output_layer = layers.Conv2DTranspose(1, (3, 3), activation='tanh', padding='same')  # Output: 64x64x1

    def call(self, inputs):
        x = self.fc(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.output_layer(x)


# Full VAE Model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = VAEDecoder(latent_dim)

    def call(self, inputs):
        frames, keyboard_inputs = inputs

        # Combine frame and keyboard input
        batch_size = tf.shape(frames)[0]
        height, width = tf.shape(frames)[1], tf.shape(frames)[2]
        num_keyboard_features = tf.shape(keyboard_inputs)[1]

        keyboard_inputs = tf.reshape(keyboard_inputs, [batch_size, 1, 1, num_keyboard_features])
        keyboard_inputs = tf.tile(keyboard_inputs, [1, height, width, 1])
        x = tf.concat([frames, keyboard_inputs], axis=-1)  # Combine along the last dimension

        # Encode to latent space
        mean, log_var = self.encoder(x)
        z = self.sampling((mean, log_var))

        # Decode to reconstruct frame
        reconstructed_frame = self.decoder(z)
        return reconstructed_frame, mean, log_var



def vae_loss(original, reconstructed, mean, log_var):
    # Ensure original and reconstructed tensors are 4D (batch, height, width, channels)
    original = tf.expand_dims(original, axis=0) if len(original.shape) == 3 else original
    reconstructed = tf.expand_dims(reconstructed, axis=0) if len(reconstructed.shape) == 3 else reconstructed

    # Reconstruction loss (pixel-wise mean squared error)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(original - reconstructed), axis=(1, 2, 3)))

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))

    # Combine the losses with a balancing coefficient
    kl_weight = 0.05  # Adjust this weight as needed
    total_loss = reconstruction_loss + kl_weight * kl_loss

    return total_loss



# Load Frames and Inputs
def load_data(image_folder, input_file):
    images = []
    inputs = []

    # Read input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, input_vector = line.strip().split(":")
        input_vector = list(map(float, input_vector.replace("[", "").replace("]", "").split(",")))

        # Load and normalize image
        image_path = os.path.join(image_folder, f"{filename}.png")
        image = Image.open(image_path).convert("L")
        image = image.resize((256, 64))  # Resize to desired dimensions
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        images.append(image)
        inputs.append(input_vector)

    return np.array(images), np.array(inputs)

def plot_reconstruction(frames, inputs):
    # Evaluate the reconstruction on sample images
    sample_indices = np.random.choice(len(frames), size=10, replace=False)
    sample_frames = frames[sample_indices]
    sample_inputs = inputs[sample_indices]

    # Reconstruct images
    reconstructed_frames, _, _ = vae((sample_frames, sample_inputs))

    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(10):
        # Original image
        plt.subplot(2, 10, i + 1)
        plt.imshow(sample_frames[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        # Reconstructed image
        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed_frames[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Reconstructed")

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

# Prepare Data
image_folder = "captured_frames"
input_file = "captured_frames/key_logs.txt"
frames, inputs = load_data(image_folder, input_file)

# Define the input shapes
frame_shape = frames.shape[1:]  # (64, 64, 1)
input_shape = inputs.shape[1:]  # (2,)

# Define the input shapes
input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]

# Calculate latent_dim as 20% of the total number of pixels
latent_dim = int(0.1 * input_height * input_width * input_channels)
print(f"Latent Dimension: {latent_dim}")

vae = VAE(latent_dim)

# Build the VAE model with specified input shapes
frame_input = tf.keras.Input(shape=frame_shape)
input_vector = tf.keras.Input(shape=input_shape)
vae((frame_input, input_vector))  # Build the model by calling it with dummy inputs

# Compile the VAE
vae.compile(optimizer='adam',
            loss=lambda y_true, y_pred: vae_loss(y_true[0], y_pred[0], y_pred[1], y_pred[2]))

# Train the VAE
history = vae.fit((frames, inputs), frames, batch_size=128, epochs=10)
plot_loss(history)
plot_reconstruction(frames, inputs)