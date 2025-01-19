import dataclasses


@dataclasses.dataclass(frozen=True)
class NNGCConstants:
    compressed_image_size=(368, 288)
    color_mode="RGB"
    latent_dimension=64
    action_weight = 1


    # sequence_length = 400
    sequence_length = 100

