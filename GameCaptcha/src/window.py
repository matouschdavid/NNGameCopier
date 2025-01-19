from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time
from pynput import keyboard

from game_capture import one_hot_encode_input
from game_utils import predict_next_frame, update_latent_space_buffer, clean_image
from plot_utils import plot_frame
import numpy as np


class Window:
    encoder_part = None
    input_part = None
    time_part = None

    def __init__(self, window_title="NNGameCopier"):
        # Create the main window
        self.root = Tk()
        self.root.title(window_title)

        # Create a label to display the image
        self.image_label = Label(self.root)
        self.image_label.pack()

        # Placeholder for the current image
        self.current_image = None
        self.input_vector = [0, 0]
        self.current_keys = set()
        self.running = True

    def set_image(self, image):
        """Sets a new image to be displayed in the window."""
        self.current_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference

    def update(self, decoder, predictor, input_vector, max_time, input_prominence, time_dim, resolution, latent_shape):
        """Updates the window with the predicted frames."""
        next_image, new_encoder_element, new_input_element, new_time_element = predict_next_frame(decoder, predictor, self.encoder_part, self.input_part, self.time_part, max_time, input_vector, input_prominence, time_dim, latent_shape, bot=False)
        self.update_buffers(new_encoder_element, new_input_element, new_time_element)

        next_image = clean_image(next_image)

        next_image_pil = Image.fromarray(next_image)
        next_image_pil = next_image_pil.resize(resolution)
        self.set_image(next_image_pil)
    
    def update_buffers(self, new_encoder_element, new_input_element, new_time_element):
        self.update_buffer(self.encoder_part, new_encoder_element)
        self.update_buffer(self.input_part, new_input_element)
        self.update_buffer(self.time_part, new_time_element)
    
    def update_buffer(self, buffer, new_element):
        # Roll the buffer to the left (remove the first element)
        buffer = np.roll(buffer, shift=-1, axis=0)
        # Add the new latent space at the end of the buffer
        buffer[-1] = new_element
        return buffer

    def start_prediction_loop(self, encoder_part, input_part, time_part, decoder, predictor, input_dim, target_frame_rate, max_time, input_prominence, time_dim, resolution, latent_shape):
        self.encoder_part = encoder_part
        self.input_part = input_part
        self.time_part = time_part

        """Run the prediction loop in a separate thread."""
        delta_time = 1 / target_frame_rate
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        while self.running:
            start_time = time.time()
            self.input_vector = one_hot_encode_input(self.current_keys)
            self.update(decoder, predictor, self.input_vector, max_time, input_prominence, time_dim, resolution, latent_shape)
            time_diff = time.time() - start_time

            if time_diff < delta_time:
                print("Computation was faster than the target frame rate.")
                time.sleep(delta_time - time_diff)

    def on_press(self, key):
        try:
            self.current_keys.add(f"{key.char}")
        except AttributeError:
            self.current_keys.add(f"{key}")

    def on_release(self, key):
        try:
            self.current_keys.remove(f"{key.char}")
        except (AttributeError, KeyError):
            self.current_keys.discard(f"{key}")

        if key == keyboard.Key.esc:
            self.running = False
            return False

    def one_hot_encode_input(keys):
        # [<space-bit>, <down-bit>]
        output = [1 if "Key.space" in keys else 0, 1 if "Key.down" in keys else 0]

        return output

    def start(self):
        """Starts the tkinter main loop."""
        self.root.mainloop()