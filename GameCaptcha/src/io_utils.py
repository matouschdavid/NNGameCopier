import os
from PIL import Image
import numpy as np

def load_data(image_folder, input_file, max=-1):
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

        if -1 < max < len(images):
            break

    return np.array(images), np.array(inputs)