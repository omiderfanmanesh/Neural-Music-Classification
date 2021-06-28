# encoding: utf-8


import os
import sys
from os import mkdir

import torch

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    num_gpus = cfg.MODEL.NUM_GPU
    device = cfg.MODEL.DEVICE
    output_dir = cfg.DIR.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    model = build_model(cfg)
    weight = torch.load(cfg.DIR.FINAL_MODEL + cfg.TEST.WEIGHT)
    model.to(device=device)
    model.load_state_dict(weight)
    val_loader = make_data_loader(cfg, inference=True)

    inference(cfg, model, val_loader)


if __name__ == '__main__':
    main()
