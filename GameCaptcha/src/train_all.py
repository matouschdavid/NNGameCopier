from GameCaptcha.src.train_auto_encoder import train_vae_main
from GameCaptcha.src.train_frame_predictor import train_prediction_main

postfix = "_flappy_64"

encoder_path = f"models/vae_encoder{postfix}.keras"
decoder_path = f"models/vae_decoder{postfix}.keras"
predictor_path = f"models/transformer_30_100_model{postfix}.keras"


def train_all():
    # train_vae_main(encoder_path, decoder_path, epochs=10, batch_size=128)
    train_prediction_main(encoder_path, decoder_path, predictor_path, epochs=30, batch_size=512)




train_all()