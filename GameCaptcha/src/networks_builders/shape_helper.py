def get_shapes(frames):
    input_height, input_width, input_channels = frames.shape[1], frames.shape[2], frames.shape[3]
    input_shape = (input_height, input_width, input_channels)  # 48 x 256 x 1
    downscale_factor = 16
    latent_height = int(input_height / downscale_factor)
    latent_width = int(input_width / downscale_factor)
    latent_channels = 64

    latent_shape = (latent_height, latent_width, latent_channels)

    return input_shape, latent_shape