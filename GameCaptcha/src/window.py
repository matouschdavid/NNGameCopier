from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time

from GameCaptcha.src.game_utils import predict_next_frame, update_latent_space_buffer, clean_image
from GameCaptcha.src.plot_utils import plot_frame


class Window:
    def __init__(self, window_title="NNGameCopier"):
        # Create the main window
        self.root = Tk()
        self.root.title(window_title)

        # Create a label to display the image
        self.image_label = Label(self.root)
        self.image_label.pack()

        # Placeholder for the current image
        self.current_image = None

    def set_image(self, image):
        """Sets a new image to be displayed in the window."""
        self.current_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference

    def update(self, latent_space_buffer, decoder, predictor, input_dim):
        """Updates the window with the predicted frames."""
        next_image, next_latent_space = predict_next_frame(decoder, predictor, latent_space_buffer, [1,0])
        latent_space_buffer = update_latent_space_buffer(latent_space_buffer, next_latent_space)

        next_image = clean_image(next_image)

        next_image_pil = Image.fromarray(next_image)
        self.set_image(next_image_pil)
        print("Updated image")

        return latent_space_buffer

    def start_prediction_loop(self, latent_space_buffer, decoder, predictor, input_dim, target_frame_rate):
        """Run the prediction loop in a separate thread."""
        delta_time = 1 / target_frame_rate
        while True:
            start_time = time.time()
            latent_space_buffer = self.update(latent_space_buffer, decoder, predictor, input_dim)
            time_diff = time.time() - start_time

            if time_diff < delta_time:
                print("Computation was faster than the target frame rate.")
                time.sleep(delta_time - time_diff)

    def start(self):
        """Starts the tkinter main loop."""
        self.root.mainloop()