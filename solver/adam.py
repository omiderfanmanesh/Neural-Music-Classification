import torch.optim as optim


class Adam:
    def __init__(self, cfg, model_params):
        self.cfg = cfg
        self.model_params = model_params

    def get_params(self):
        return {
            'params': self.model_params,
            'lr': self.cfg.OPT.ADAM.LR,
            'betas': self.cfg.OPT.ADAM.BETAS,
            'eps': self.cfg.OPT.ADAM.EPS,
            'weight_decay': self.cfg.OPT.ADAM.WEIGHT_DECAY,
            'amsgrad': self.cfg.OPT.ADAM.AMS_GRAD
        }

    def optimizer(self):
        params = self.get_params()
        return optim.Adam(**params)
