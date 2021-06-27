# encoding: utf-8


from time import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from utils import AverageMeter
from utils import utilities


def do_train(
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
            torch.save(model.state_dict(), 'saved_model.pth')

    print('Finished Training')

#
# def do_train(
#         cfg,
#         model,
#         train_loader,
#         val_loader,
#         optimizer,
#         scheduler,
#         loss_fn,
# ):
#     model_name = cfg.MODEL.NAME
#     log_period = cfg.SOLVER.LOG_PERIOD
#     checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
#     output_dir = cfg.DIR.OUTPUT_DIR
#     device = cfg.MODEL.DEVICE
#     device = torch.device(device=device)
#     epochs = cfg.SOLVER.MAX_EPOCHS
#
#     model = model.to(device)
#
#     logger = logging.getLogger("template_model.train")
#     logger.info("Start training")
#     # Create a logger
#
#     precision = Precision(average=False)
#     recall = Recall(average=False)
#     # F1 = (precision * recall * 2 / (precision + recall)).mean()
#     F1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)
#
#     trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
#     evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
#                                                             'precision': precision,
#                                                             'recall': recall,
#                                                             'f1': F1,
#                                                             'ce_loss': Loss(loss_fn)}, device=device)
#
#     checkpointer = ModelCheckpoint(output_dir, model_name, n_saved=5, require_empty=False)
#     timer = Timer(average=True)
#
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
#                                                                      'optimizer': optimizer})
#
#     timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
#                  pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
#
#     RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')
#
#     pbar = ProgressBar(persist=True, bar_format="")
#     pbar.attach(trainer)
#
#
#     def score_function(engine):
#         val_loss = engine.state.metrics['ce_loss']
#         return -1 * val_loss
#
#     early_stopping_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
#     # evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
#
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def log_training_loss(engine):
#         iter = (engine.state.iteration - 1) % len(train_loader) + 1
#
#         if iter % log_period == 0:
#             logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
#                         .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))
#
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_training_results(engine):
#         evaluator.run(train_loader)
#
#         metrics = evaluator.state.metrics
#         _avg_accuracy = metrics['accuracy']
#
#         _precision = metrics['precision']
#         _precision = torch.mean(_precision)
#
#         _recall = metrics['recall']
#         _recall = torch.mean(_recall)
#
#         _f1 = metrics['f1']
#         _f1 = torch.mean(_f1)
#
#         _avg_loss = metrics['ce_loss']
#
#         logger.info(
#             "Training Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f},  "
#             "Avg Loss: {:.3f} ".format(engine.state.epoch, _avg_accuracy, _precision, _recall, _f1, _avg_loss))
#
#     if val_loader is not None:
#         @trainer.on(Events.EPOCH_COMPLETED)
#         def log_validation_results(engine):
#             evaluator.run(val_loader)
#             metrics = evaluator.state.metrics
#
#             _avg_accuracy = metrics['accuracy']
#
#             _precision = metrics['precision']
#             _precision = torch.mean(_precision)
#
#             _recall = metrics['recall']
#             _recall = torch.mean(_recall)
#
#             _f1 = metrics['f1']
#             _f1 = torch.mean(_f1)
#
#             _avg_loss = metrics['ce_loss']
#
#             _avg_loss = metrics['ce_loss']
#             logger.info(
#                 "Validation Results - Epoch: {} Avg accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}, Avg Loss: {:.3f}"
#                     .format(engine.state.epoch, _avg_accuracy, _precision, _recall, _f1, _avg_loss)
#             )
#
#     # adding handlers using `trainer.on` decorator API
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def print_times(engine):
#         logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
#                     .format(engine.state.epoch, timer.value() * timer.step_count,
#                             train_loader.batch_size / timer.value()))
#         timer.reset()
#
#     def get_saved_model_path(epoch):
#         return f'{cfg.DIR.BEST_MODEL}/Model_{model_name}_{epoch}.pth'
#
#     # best_loss = 0.
#     # best_epoch = 1
#     # best_epoch_file = ''
#     #
#     # @trainer.on(Events.EPOCH_COMPLETED)
#     # def save_best_epoch_only(engine):
#     #     epoch = engine.state.epoch
#     #
#     #     global best_loss
#     #     global best_epoch
#     #     global best_epoch_file
#     #     best_loss = 0. if epoch == 1 else best_loss
#     #     best_epoch = 1 if epoch == 1 else best_epoch
#     #     best_epoch_file = '' if epoch == 1 else best_epoch_file
#     #     metrics = evaluator.run(val_loader).metrics
#     #     if metrics['ce_loss'] < best_loss:
#     #         prev_best_epoch_file = get_saved_model_path(best_epoch)
#     #         if os.path.exists(prev_best_epoch_file):
#     #             os.remove(prev_best_epoch_file)
#     #
#     #         best_loss = metrics['ce_loss']
#     #         best_epoch = epoch
#     #         best_epoch_file = get_saved_model_path(best_epoch)
#     #         print(f'\nEpoch: {best_epoch} - Loss is improved! Loss: {best_loss}\n\n\n')
#     #         torch.save(model, best_epoch_file)
#
#     trainer.run(train_loader, max_epochs=epochs)
#     torch.save(model.state_dict(), cfg.DIR.FINAL_MODEL + '/final_artist20_slice_3s_model_state_dic.pt')
#     torch.save(model, cfg.DIR.FINAL_MODEL + '/final_artist20_slice_3s_model.pt')
