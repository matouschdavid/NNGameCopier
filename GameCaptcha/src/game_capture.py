import os
import time
from pynput import keyboard
import pyautogui
from PIL import ImageGrab
from threading import Thread
from screeninfo import get_monitors
from pynput.mouse import Controller

monitor = get_monitors()[0]
print(monitor.width, monitor.height)
mouse = Controller()

width = 600
height = width / 4
start_x = monitor.width / 2 - width / 2
start_y = 180

# Folder to save screenshots
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store key logs for each frame
frame_logs = {}
running = True
capture_started = False

current_keys = set()

def process_setup_keys(key):
    global start_x, start_y, width, height, capture_started
    k_chr = key.char
    mouse_position = mouse.position

    if k_chr == "1":
        start_x = mouse_position[0]
        start_y = mouse_position[1]
        print(start_x, start_y)
    elif k_chr == "2":
        end_x = mouse_position[0]
        end_y = mouse_position[1]

        width = end_x - start_x
        height = end_y - start_y
        print(width, height)
    elif k_chr == "3":
        capture_started = True
        print("capture_started")

def on_press(key):
    try:
        current_keys.add(f"{key.char}")
        process_setup_keys(key)
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

        if capture_started:
            # Capture the screen
            screenshot = ImageGrab.grab(bbox=(start_x, start_y, start_x + width, start_y + height))  # Adjust the coordinates as needed
            screenshot.save(os.path.join(output_folder, f"frame_{frame_count:010}.png"))

            # Log the current keys for this frame
            frame_logs[frame_count] = one_hot_encode_input(current_keys)

            frame_count += 1

        time.sleep(max(0, int((1 / 60) - (time.time() - frame_time))))

def one_hot_encode_input(keys):
    # [<space-bit>, <down-bit>]
    output = [1 if "Key.space" in keys else 0, 1 if "Key.down" in keys else 0]

    return output

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
            f.write(f"frame_{frame:010}: {keys}\n")

if __name__ == "__main__":
    main()
