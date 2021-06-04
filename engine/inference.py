import logging

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Fbeta
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall


def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'precision': precision,
                                                            'recall': recall,
                                                            'f1': F1}, device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_acc = metrics['accuracy']
        logger.info("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    evaluator.run(val_loader)
