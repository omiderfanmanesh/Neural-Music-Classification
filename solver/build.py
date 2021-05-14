from .optimizer import OptimizerFactory


def make_optimizer(cfg, model_params, opt):
    op = OptimizerFactory(cfg=cfg, model_params=model_params, opt=opt).get_opt()
    return op
