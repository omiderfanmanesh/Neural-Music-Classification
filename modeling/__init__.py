# encoding: utf-8

from .model import MusicClassification


def build_model(cfg):
    model = MusicClassification(cfg)
    return model
