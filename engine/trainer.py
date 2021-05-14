# encoding: utf-8


import logging

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage, Recall, Precision
from ignite.metrics.metrics_lambda import MetricsLambda


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    device = torch.device("cuda")
    epochs = cfg.SOLVER.MAX_EPOCHS

    model = model.to(device)

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")

    precision = Precision(average=True, is_multilabel=True)
    recall = Recall(average=True, is_multilabel=True)
    F1 = precision * recall * 2 / (precision + recall + 1e-20)
    F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'precision': precision,
                                                            'recall': recall,
                                                            'f1': F1,
                                                            'ce_loss': Loss(loss_fn)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'music', n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']

        avg_loss = metrics['ce_loss']

        logger.info(
            "Training Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f},  Avg Loss: {:.3f}"
                .format(engine.state.epoch, avg_accuracy, precision, recall, f1, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics

            avg_accuracy = metrics['accuracy']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']

            avg_loss = metrics['ce_loss']
            logger.info(
                "Validation Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}, Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_accuracy, precision, recall, f1, avg_loss)
            )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
