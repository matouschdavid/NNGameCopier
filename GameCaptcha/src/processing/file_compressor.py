import os
import shutil
from PIL import Image

import GameCaptcha.src.config as config
from GameCaptcha.src.util.io_utils import extract_data_from_line


# Load Frames and Inputs
def compress_data():
    input_file = f"{config.captured_folder}/key_logs.txt"

    os.makedirs(config.compressed_folder, exist_ok=True)
    shutil.copy(input_file, os.path.join(config.compressed_folder, os.path.basename(input_file)))
    # Read input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, _, _ = extract_data_from_line(line.strip())
        # Load and normalize image
        image_path = os.path.join(config.captured_folder, f"{filename}.png")
        image = Image.open(image_path).convert(config.frame_channels)
        image = image.resize(config.compressed_frame_resolution)  # Resize to desired dimensions
        image.save(os.path.join(config.compressed_folder, f"{filename}.png"))


# Prepare Data
compress_data()