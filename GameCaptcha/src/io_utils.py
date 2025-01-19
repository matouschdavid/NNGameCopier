import os
from PIL import Image
import numpy as np
import re

from tensorflow import keras

from GameCaptcha.src.constants import NNGCConstants


def load_data(image_folder, input_file, min=0, max=-1):
    images = []
    inputs = []
    timestamps = []

    # Read input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, input_vector, time_stamp = extract_data_from_line(line.strip())

        if min == 0 or int(filename.split("_")[1]) >= min:
            # Load and normalize image
            image_path = os.path.join(image_folder, f"{filename}.png")
            image = Image.open(image_path).convert(NNGCConstants.color_mode)
            image = image.resize(NNGCConstants.compressed_image_size)  # Resize to desired dimensions
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=-1)  # Add channel dimension

            images.append(image)
            inputs.append(input_vector)
            timestamps.append(time_stamp)

            if -1 < max < len(images):
                break

    return np.array(images), np.array(inputs), np.array(timestamps)

def extract_data_from_line(line):
    pattern = r"(frame_\d+): \(\[([^\]]+)\], (\d+)\)"
    match = re.match(pattern, line)

    frame = match.group(1)
    input_vector = [int(x) for x in match.group(2).split(",")]
    time_stamp = int(match.group(3))

    return frame, input_vector, time_stamp

class ImageDataGenerator(keras.utils.Sequence):
    def __init__(self, image_folder, input_file, batch_size, min=0, max=-1):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.min = min
        self.max = max

        # Read file paths once
        with open(input_file, "r") as f:
            self.lines = f.readlines()
            if max > 0:
                self.lines = self.lines[:max]

        self.n_samples = len(self.lines)
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []

        for i in batch_indices:
            line = self.lines[i]
            filename, _, _ = extract_data_from_line(line.strip())

            image_path = os.path.join(self.image_folder, f"{filename}.png")
            image = Image.open(image_path).convert(NNGCConstants.color_mode)
            image = image.resize(NNGCConstants.compressed_image_size)
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=-1)

            batch_images.append(image)

        X, y = np.array(batch_images), np.array(batch_images)

        return X, y

class ImageDataGeneratorEager(keras.utils.Sequence):
    def __init__(self, image_folder, input_file, batch_size, min=0, max=-1):
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.min = min
        self.max = max

        # Read file paths once
        with open(input_file, "r") as f:
            self.lines = f.readlines()
            if max > 0:
                self.lines = self.lines[:max]

        self.n_samples = len(self.lines)
        self.indices = np.arange(self.n_samples)

        # Pre-load all images
        self.images = self._load_all_images()

    def _load_all_images(self):
        """Load all images into memory during initialization"""
        images = []
        for line in self.lines:
            filename, _, _ = extract_data_from_line(line.strip())
            image_path = os.path.join(self.image_folder, f"{filename}.png")

            # Load and preprocess image
            image = Image.open(image_path).convert(NNGCConstants.color_mode)
            image = image.resize(NNGCConstants.compressed_image_size)
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=-1)

            images.append(image)

        return np.array(images)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Simply index into pre-loaded images
        batch_images = self.images[batch_indices]

        return batch_images, batch_images

class LSTMImageDataGeneratorEager(keras.utils.Sequence):
    def __init__(self, image_folder, input_file, batch_size, sequence_length, encoder, min=0, max=-1):
        # super().__init__([], {})
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.min = min
        self.max = max
        self.encoder = encoder

        # Initialize cache
        self.encoded_image_cache = {}

        # Read file paths and data
        with open(input_file, "r") as f:
            self.lines = f.readlines()
            if max > 0:
                self.lines = self.lines[:max]

        # Calculate valid sequences
        self.n_samples = len(self.lines) - sequence_length
        self.indices = np.arange(self.n_samples)

        # Preload all images
        self._preload_images()

    def _preload_images(self):
        print("Preloading and encoding images...")
        unique_filenames = set()

        # Collect unique filenames
        for line in self.lines:
            filename, _, _ = extract_data_from_line(line.strip())
            unique_filenames.add(filename)

        # Load and encode unique images in batches
        batch_size = 32  # Adjust based on your memory constraints
        filenames = list(unique_filenames)

        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i + batch_size]
            batch_images = []

            for filename in batch_filenames:
                image_path = os.path.join(self.image_folder, f"{filename}.png")
                image = Image.open(image_path).convert(NNGCConstants.color_mode)
                image = image.resize(NNGCConstants.compressed_image_size)
                image = np.array(image) / 255.0
                image = np.expand_dims(image, axis=0)
                batch_images.append(image)

            # Encode batch
            batch_images = np.vstack(batch_images)
            _, _, encoded_batch = self.encoder(batch_images)

            # Store in cache
            for idx, filename in enumerate(batch_filenames):
                self.encoded_image_cache[filename] = encoded_batch[idx]

            if (i + batch_size) % 1000 == 0:
                print(f"Processed {i + batch_size}/{len(filenames)} images")

        print("Finished preloading images")

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_sequences_images = []
        batch_sequences_inputs = []
        batch_sequences_timestamps = []

        for i in batch_indices:
            sequence_images = []
            sequence_inputs = []
            sequence_timestamps = []

            for j in range(self.sequence_length + 1):
                line = self.lines[i + j]
                filename, input_vector, timestamp = extract_data_from_line(line.strip())

                # Get encoded image from cache
                encoded_image = self.encoded_image_cache[filename]

                sequence_images.append(encoded_image)
                sequence_inputs.append(input_vector * NNGCConstants.action_weight)
                sequence_timestamps.append(timestamp)

            batch_sequences_images.append(sequence_images)
            batch_sequences_inputs.append(sequence_inputs)
            batch_sequences_timestamps.append(sequence_timestamps)

        # batch_sequences_images_head = batch_sequences_images[:][:-1].copy()


        batch_sequences_images_head = np.array([seq_position[:-1] for seq_position in batch_sequences_images])
        batch_sequences_inputs = np.array([seq_position[:-1] for seq_position in batch_sequences_inputs])
        # batch_sequences_inputs = np.array(batch_sequences_inputs[:][:-1])
        batch_sequences_timestamps = np.array(batch_sequences_timestamps)

        # Return format matches prepare_training_data
        X = (batch_sequences_images_head, batch_sequences_inputs)
        y = np.array([seq_position[-1] for seq_position in batch_sequences_images])
        # print(X.shape)
        # print(y.shape)
        return X, y

    def clear_cache(self):
        """Clear the image cache to free memory if needed"""
        self.encoded_image_cache.clear()