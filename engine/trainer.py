# encoding: utf-8

import logging
from time import time

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage, Fbeta
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from sklearn.metrics import f1_score

from metrics.f1score import F1Score
from utils import AverageMeter
from utils import utilities
from utils.utilities import log_result
from utils.utilities import tesnorboard


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
):
    # train by using pytorch-ignite
    train_ignite(cfg=cfg,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 criterion=criterion)

    # train without library
    # train(cfg=cfg,
    #       model=model,
    #       train_loader=train_loader,
    #       val_loader=val_loader,
    #       optimizer=optimizer,
    #       scheduler=scheduler,
    #       criterion=criterion)


def train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
):
    model_name = cfg.MODEL.NAME
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.DIR.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    device = torch.device(device=device)
    epochs = cfg.SOLVER.MAX_EPOCHS

    model = model.to(device)

    min_valid_loss = np.inf

    for e in range(epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        train_losses = AverageMeter('Training Loss', ':.4e')
        val_losses = AverageMeter('Validation Loss', ':.4e')
        train_accuracy = AverageMeter('Training Accuracy', ':6.2f')
        val_accuracy = AverageMeter('Validation Accuracy', ':6.2f')
        train_f1 = AverageMeter('Training F1 score', ':6.2f')
        val_f1 = AverageMeter('Validation F1 score', ':6.2f')

        end = time()
        for itr, (data, labels) in enumerate(train_loader):
            data_time.update(time() - end)
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            # train_loss = loss.item() * data.size(0)
            # accuracy
            acc = utilities.accuracy(y_true=labels, y_pred=target)
            # print(acc)
            _, predicted = torch.max(target.data, 1)
            f = f1_score(labels.cpu(), predicted.cpu(), average='micro')
            # print(f)
            train_f1.update(f)
            train_accuracy.update(acc)
            train_losses.update(loss.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time() - end)
            end = time()
            utilities.progress_bar(current=itr, total=len(train_loader))

        print('Validating...')
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in val_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate Loss
            # valid_loss = loss.item() * data.size(0)
            val_losses.update(loss.item(), data.size(0))

            acc = utilities.accuracy(y_true=labels, y_pred=target)
            # print(f"acc{acc}")
            # accuracy
            _, predicted = torch.max(target.data, 1)
            val_accuracy.update(acc)
            f = f1_score(labels.cpu(), predicted.cpu(), average='micro')
            # print(f"f{f}")
            val_f1.update(f)

        model.train()

        print(
            f'Epoch {e + 1} [{data_time.avg:.2f}s]\t\t Training Loss: {train_losses.avg:.2f} \t\t '
            f'Validation Loss: {val_losses.avg:.2f}, Train Accuracy: {train_accuracy.avg:.2f}, Train f1 score: {train_f1.avg:.2f},'
            f' Validation Accuracy: {val_accuracy.avg:.2f}, Validation f1 score: {val_f1.avg:.2f}')

        if min_valid_loss > val_losses.avg:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_losses.avg:.6f}) \t Saving The Model')
            min_valid_loss = val_losses.avg

            # Saving State Dict
            torch.save(model.state_dict(), cfg.DIR.BEST_MODEL + cfg.TEST.WEIGHT)

    print('Finished Training')


def train_ignite(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
):
    model_name = cfg.MODEL.NAME
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.DIR.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    device = torch.device(device=device)
    epochs = cfg.SOLVER.MAX_EPOCHS

    model = model.to(device)

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    # Create a logger

    precision = Precision(average=False)
    recall = Recall(average=False)
    # F1 = (precision * recall * 2 / (precision + recall)).mean()
    F1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'precision': precision,
                                                            'recall': recall,
                                                            'f1_micro': F1Score(),
                                                            'f1': F1,
                                                            'ce_loss': Loss(criterion)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, model_name, n_saved=5, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer)

    def score_function(engine):
        val_loss = engine.state.metrics['ce_loss']
        return -1 * val_loss

    early_stopping_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

    # evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

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
        log_result(title="Training", logger=logger, engine=engine, metrics=metrics)

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            log_result(title="Validation", logger=logger, engine=engine, metrics=metrics)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)

    tesnorboard(Events=Events, model=model, metrics=['accuracy', 'precision', 'recall', 'f1', 'f1_micro', 'ce_loss'],
                optimizer=optimizer, trainer=trainer, evaluator=evaluator,
                log_dir=cfg.DIR.TENSORBOARD_LOG)

    torch.save(model.state_dict(), cfg.DIR.FINAL_MODEL + '/30s_gtzan_no_aug_model.pt')
    torch.save(model, cfg.DIR.FINAL_MODEL + '/30s_gtzan_no_aug_model.pt')
