from .adam import Adam
from .sgd import SGD


class OptimizerFactory:
    def __init__(self, cfg, model_params, opt):
        self.cfg = cfg
        self.opt = opt
        self.model_params = model_params

    def get_opt(self):
        if str(self.opt).upper() == 'ADAM':
            return Adam(self.cfg, self.model_params).optimizer()
        elif str(self.opt).upper() == 'SGD':
            return SGD(self.cfg, self.model_params).optimizer()
        else:
            raise ValueError(self.opt)
