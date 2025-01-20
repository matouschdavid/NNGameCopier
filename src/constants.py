import dataclasses


@dataclasses.dataclass(frozen=True)
class NNGCConstants:
    latent_dimension=64
    action_weight = 1
    # sequence_length = 400
    sequence_length = 100

    # DINO METADATA
    # compressed_image_size=(256, 48) # dino
    # color_mode="L" # dino
    # action_count = 2 # dino
    # model_path = "models_dino/"
    # image_path = "compressed_frames_dino"
    # input_file = f"{image_path}/key_logs.txt"
    # postfix = "_dino_64"
    # plot_title = "Sequence Prediction Dino Game"

    # FLAPPY METADATA
    compressed_image_size=(368, 288) # flappy
    color_mode="RGB" # flappy
    action_count = 1 # dino
    model_path = "models_flappy/"
    image_path = "compressed_frames_flappy"
    input_file = f"{image_path}/key_logs.txt"
    postfix = "_flappy_64"
    plot_title = "Sequence Prediction Flappy Game"
