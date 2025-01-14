import os
from PIL import Image
import numpy as np
import re

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
            image = Image.open(image_path).convert("L")
            image = image.resize((256, 64))  # Resize to desired dimensions
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