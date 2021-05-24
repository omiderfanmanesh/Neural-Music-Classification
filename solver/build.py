from .optimizer import OptimizerFactory


def make_optimizer(cfg, model_params, opt):
    op = OptimizerFactory(cfg=cfg, model_params=model_params, opt=opt)
    optimizer = op.get_opt()
    return optimizer
