import os
import time
from pynput import keyboard
import pyautogui
from PIL import ImageGrab
from threading import Thread

# Folder to save screenshots
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store key logs for each frame
frame_logs = {}
running = True

current_keys = set()

def on_press(key):
    try:
        current_keys.add(f"{key.char}")
    except AttributeError:
        current_keys.add(f"{key}")

def on_release(key):
    global running
    try:
        current_keys.remove(f"{key.char}")
    except (AttributeError, KeyError):
        current_keys.discard(f"{key}")

    if key == keyboard.Key.esc:
        running = False
        return False

def capture_screen():
    frame_count = 0
    start_time = time.time()
    while running:
        frame_time = time.time() - start_time

        # Capture the screen
        screenshot = ImageGrab.grab(bbox=(100, 100, 800, 600))  # Adjust the coordinates as needed
        screenshot.save(os.path.join(output_folder, f"frame_{frame_count:05}.png"))

        # Log the current keys for this frame
        frame_logs[frame_count] = list(current_keys)

        frame_count += 1
        time.sleep(max(0, int((1 / 60) - (time.time() - frame_time))))

def main():
    # Start the key listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Start screen capturing
    capture_thread = Thread(target=capture_screen)
    capture_thread.start()

    # Wait for threads to finish
    listener.join()
    capture_thread.join()

    # Save key logs
    with open(os.path.join(output_folder, "key_logs.txt"), "w") as f:
        for frame, keys in sorted(frame_logs.items()):
            f.write(f"Frame {frame:05}: {', '.join(keys) if keys else 'No keys'}\n")

if __name__ == "__main__":
    main()
