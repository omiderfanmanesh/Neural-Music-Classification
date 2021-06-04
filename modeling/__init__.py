# encoding: utf-8
import torch

from .fine_tuned_model import FineTuneModel
from .model import MusicClassificationCRNN, add_noise_to_weights


def build_model(cfg):
    is_pretrained = cfg.MODEL.PRE_TRAINED
    pre_trained_path = cfg.MODEL.PRE_TRAINED_ADDRESS
    model = MusicClassificationCRNN(cfg)
    if is_pretrained:
        org_model = torch.load(pre_trained_path)
        org_model.pop(key='dense.weight')
        org_model.pop(key='dense.bias')
        model.load_state_dict(org_model, strict=False)

    # model.apply(add_noise_to_weights)
    return model
