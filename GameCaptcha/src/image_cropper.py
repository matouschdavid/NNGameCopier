from PIL import Image
import os

# Folder containing the PNG files
folder_path = "captured_frames"
crop_height = 30  # Height in pixels to crop from the top

# Ensure the path exists
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
    exit()

# Process each PNG file in the folder
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".png"):
        file_path = os.path.join(folder_path, file_name)

        # Open the image
        with Image.open(file_path) as img:
            # Define cropping box (left, upper, right, lower)
            crop_box = (0, crop_height, img.width, img.height)
            cropped_img = img.crop(crop_box)

            # Overwrite the file
            cropped_img.save(file_path)

print("All PNG files have been cropped and overwritten.")
