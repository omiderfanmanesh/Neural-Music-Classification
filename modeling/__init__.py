# encoding: utf-8

from .model import MusicClassificationCRNN, add_noise_to_weights


def build_model(cfg):
    model = MusicClassificationCRNN(cfg)
    # model.apply(add_noise_to_weights)
    return model
