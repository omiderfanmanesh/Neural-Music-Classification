import torch.optim as optim


class SGD:
    def __init__(self, cfg, model_params):
        self.cfg = cfg
        self.model_params = model_params

    def get_params(self):
        return {
            'params': self.model_params,
            'lr': self.cfg.OPT.SGD.LR,
            'momentum': self.cfg.OPT.SGD.MOMENTUM,
            'weight_decay': self.cfg.OPT.SGD.WEIGHT_DECAY,
            'dampening': self.cfg.OPT.SGD.DAMPENING,
            'nesterov': self.cfg.OPT.SGD.NESTEROV
        }

    def optimizer(self):
        params = self.get_params()
        return optim.SGD(**params)
