import dataclasses


@dataclasses.dataclass(frozen=True)
class NNGCConstants:
    compressed_image_size=(256, 48)
    color_mode="L"
    latent_dimension=64
    action_weight = 1
    action_count = 2


    # sequence_length = 400
    sequence_length = 100

