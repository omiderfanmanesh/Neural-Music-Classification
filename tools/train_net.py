# encoding: utf-8


import os
import sys
from os import mkdir

import torch.nn.functional as F

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger


def train(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    arguments = {}

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    criterion = F.cross_entropy

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

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    train(cfg)


if __name__ == '__main__':
    main()
