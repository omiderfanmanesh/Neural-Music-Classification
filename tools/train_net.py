# encoding: utf-8


import os
import sys
from os import mkdir

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger

import random

SEED = 2021
random.seed(SEED)

import torch
import torch.nn as nn
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.empty_cache()
import numpy as np

np.random.seed(SEED)


def train(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg=cfg,
                               model_params=model.parameters(),
                               opt='ADAM')
    scheduler = None

    arguments = {}

    train_loader, test_loader, val_loader = make_data_loader(cfg)

    criterion = nn.CrossEntropyLoss()

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        None,
        criterion
    )


def main():
    num_gpus = 1

    output_dir = cfg.DIR.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    best_models = cfg.DIR.BEST_MODEL
    if best_models and not os.path.exists(best_models):
        mkdir(best_models)

    tensorboard_log = cfg.DIR.TENSORBOARD_LOG
    if tensorboard_log and not os.path.exists(tensorboard_log):
        mkdir(tensorboard_log)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    train(cfg)


if __name__ == '__main__':
    main()
