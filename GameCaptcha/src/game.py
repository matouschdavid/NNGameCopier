import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import threading
import time
from collections import deque

from GameCaptcha.src.constants import NNGCConstants
from GameCaptcha.src.io_utils import load_data
from GameCaptcha.src.train_frame_predictor import predict_next_frame, PositionalEncoding
from GameCaptcha.src.vae import Sampling

frame_with = NNGCConstants.compressed_image_size[0]
frame_height = NNGCConstants.compressed_image_size[1]
SEQUENCE_LENGTH = NNGCConstants.sequence_length  # Match your model's sequence length

class Window:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Game Prediction")

        # Create canvas for displaying frames
        self.canvas = tk.Canvas(self.root, width=frame_with, height=frame_height)
        self.canvas.pack()

        # Initialize key press state
        self.key_pressed = 0
        self.running = True

        # Initialize sequences
        self.encoded_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.action_sequence = deque(maxlen=SEQUENCE_LENGTH)

        # Bind keyboard events
        self.root.bind('<space>', self.on_space_press)
        self.root.bind('<KeyRelease-space>', self.on_space_release)

        # Initialize image display
        self.image_on_canvas = None

    def on_space_press(self, event):
        self.key_pressed = 1

    def on_space_release(self, event):
        self.key_pressed = 0

    def update_frame(self, frame):
        # Convert numpy array to PhotoImage
        image = Image.fromarray((frame * 255).astype(np.uint8))
        photo = ImageTk.PhotoImage(image)

        # Update canvas
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(frame_with//2, frame_height//2, image=photo)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=photo)

        # Keep a reference to avoid garbage collection
        self.canvas.photo = photo

    def initialize_sequences(self, initial_encoded_frames):
        # Initialize encoded_sequence with initial frames
        for frame in initial_encoded_frames:
            self.encoded_sequence.append(frame)

        # Initialize action_sequence with zeros
        for _ in range(SEQUENCE_LENGTH):
            self.action_sequence.append(0)

    def start_prediction_loop(self, decoder, predictor, frame_rate):
        while self.running:
            start_time = time.time()

            # Convert sequences to numpy arrays
            encoded_array = np.array(list(self.encoded_sequence))
            action_array = np.array([[action] for action in list(self.action_sequence)])

            # Predict next frame
            next_encoded = predict_next_frame(
                predictor,
                encoded_array,
                action_array
            )

            # Decode the frame
            next_frame = decoder.predict(next_encoded.reshape(1, -1), verbose=0)[0]

            # Update the display
            self.update_frame(next_frame)

            # Update sequences
            self.encoded_sequence.append(next_encoded)
            self.action_sequence.append(self.key_pressed)

            # Maintain frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 1/frame_rate - elapsed_time)
            time.sleep(sleep_time)

    def start(self):
        self.root.mainloop()
        self.running = False

    def stop(self):
        self.running = False
        self.root.quit()

def main():
    postfix = "_flappy_128"

    encoder_path = f"models/vae_encoder{postfix}.keras"
    decoder_path = f"models/vae_decoder{postfix}.keras"
    predictor_path = f"models/bilstm_50_model{postfix}.keras"

    # Load models and data
    encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
    decoder = load_model(decoder_path)
    predictor = load_model(predictor_path, custom_objects={'PositionalEncoding': PositionalEncoding}, safe_mode=False)

    # Load initial frames
    image_folder = "compressed_frames"
    input_file = "compressed_frames/key_logs.txt"
    frames, inputs, _ = load_data(image_folder, input_file)

    entrypoint = 170
    # Get initial encoded frames
    initial_frames = frames[entrypoint:(SEQUENCE_LENGTH+entrypoint)]
    _, _, initial_encoded = encoder(initial_frames)
    initial_encoded = initial_encoded.numpy()

    # Start the application
    frame_rate = 20
    app = Window()

    # Initialize sequences
    app.initialize_sequences(initial_encoded)

    # Start prediction thread
    prediction_thread = threading.Thread(
        target=app.start_prediction_loop,
        args=(decoder, predictor, frame_rate

              )
    )
    prediction_thread.daemon = True
    prediction_thread.start()

    # Start main loop
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()

if __name__ == "__main__":
    main()