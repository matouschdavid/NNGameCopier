import os
import shutil
from PIL import Image

from GameCaptcha.src.io_utils import extract_data_from_line


# Load Frames and Inputs
def compress_data(image_folder, input_file):
    output_folder = "compressed_frames"
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(input_file, os.path.join(output_folder, os.path.basename(input_file)))
    # Read input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, input_vector, time_stamp = extract_data_from_line(line.strip())
        # Load and normalize image
        image_path = os.path.join(image_folder, f"{filename}.png")
        image = Image.open(image_path).convert("L")
        image = image.resize((256, 48))  # Resize to desired dimensions
        image.save(os.path.join(output_folder, f"{filename}.png"))


# Prepare Data
image_folder = "captured_frames"
input_file = "captured_frames/key_logs.txt"
compress_data(image_folder, input_file)