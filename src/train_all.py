from src.constants import NNGCConstants
from src.train_auto_encoder import train_vae_main
from src.train_frame_predictor import train_prediction_main

postfix = NNGCConstants.postfix

encoder_path = f"{NNGCConstants.model_path}vae_encoder{postfix}.keras"
decoder_path = f"{NNGCConstants.model_path}vae_decoder{postfix}.keras"
predictor_path = f"{NNGCConstants.model_path}lstm_model{postfix}.keras"


def train_all():
    train_vae_main(encoder_path, decoder_path, epochs=100, batch_size=64)
    train_prediction_main(encoder_path, decoder_path, predictor_path, epochs=50, batch_size=512)

train_all()