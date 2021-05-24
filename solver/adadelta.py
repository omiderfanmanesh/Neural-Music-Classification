import torch.optim as optim


class Adadelta:
    def __init__(self, cfg, model_params):
        self.cfg = cfg
        self.model_params = model_params

    def get_params(self):
        return {
            'params': self.model_params,
            'lr': self.cfg.OPT.ADADELTA.LR,
            'rho': self.cfg.OPT.ADADELTA.MOMENTUM,
            'eps': self.cfg.OPT.ADADELTA.WEIGHT_DECAY,
            'weight_decay': self.cfg.OPT.ADADELTA.DAMPENING,
        }

    def optimizer(self):
        params = self.get_params()
        return optim.SGD(**params)
