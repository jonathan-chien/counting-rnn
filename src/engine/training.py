from contextlib import contextmanager
from itertools import chain

import torch

from . import eval, utils
from general_utils import recursion as recursion_utils
from general_utils import tensor as tensor_utils


@contextmanager
def temporarily_eval(model):
    """ 
    Temporarily place model in eval mode for evaluation on validation set during training.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def train(
    model, 
    dataloader, 
    loss_terms,
    evaluation,
    h_0=None,
    logger_train=None,
    criteria={'accuracy' : utils.compute_accuracy},
    compute_mean_for=None,
    # save_validation_logger=True,
    metric_tracker=None,
    early_stopping=None, 
    num_epochs=50, 
    device='cuda',
    deterministic=True,   
):
    """ 
    Training logic.

    Parameters
    ----------

    Returns
    -------
    """
    h_0 = utils.validate_h_0_config(h_0)

    if (
        metric_tracker and early_stopping
        and metric_tracker.metric_name != early_stopping.metric_name
    ):
        raise RuntimeError(
            "If a MetricTracker object and an EarlyStopping object are both "
            "passed in, their attributes `metric_name` must match, but got "
            f"{metric_tracker.metric_name} and {early_stopping.metric_name}, respectively."
        )

    model.to(device)
    model.train()
    if model.tokens.device != device:
        raise RuntimeError(
            f"model.tokens must be on user specified device {device} but is " 
            f"on {model.tokens.device}."
        )
    
    # If logger objects print, add demarcation before first epoch.
    if logger_train.verbose_epoch or logger_val.verbose_epoch:
        print('----------------------------------------')

    for i_epoch in range(num_epochs):
        if logger_train: epoch_log = {}

        for i_batch, (batch, labels, lengths, masks, seq_ind) in enumerate(dataloader):
            batch, labels, masks = batch.to(device), labels.to(device), masks.to(device)

            if h_0 is None:
                h_0_batch = None
            elif callable(h_0):
                h_0_batch = h_0(batch.shape[0], model)
            elif isinstance(h_0, torch.Tensor):
                if h_0.requires_grad:
                    raise RuntimeError(
                        "h_0 has requires_grad=True, but this is currently "
                        "unintended and unsupported behavior."
                    )
                h_0_batch = h_0
            else:
                raise TypeError(f"Unsupported type {type(h_0)} for `h_0`.")
            
            # Forward pass.
            logits, _, _ = model(
                batch, h_0=h_0_batch, lengths=lengths, output_type='many_to_many'
            )
            _, pred_labels = model.to_token(logits, deterministic=deterministic)

            # Shift masks forward by one since prediction is of next token.
            shifted_masks = torch.roll(masks, shifts=-1, dims=1)

            # Compute losses.
            losses = {
                loss_term.name : loss_term.compute_loss(
                    logits[shifted_masks], labels[masks], model
                )
                for loss_term in loss_terms
            }
            for loss_term in loss_terms: 
                loss_term.step()

            # Compute performance metrics.
            if criteria is not None:
                performance = {
                    criterion_name : criterion(
                        pred_labels,
                        labels.roll(-1, dims=1),
                        masks.roll(-1, dims=1)
                    )
                    for criterion_name, criterion in criteria.items()
                }

            # Optionally log losses/performance metrics.
            if logger_train: 
                batch_log = {'batch_size' : batch.shape[0]}
                batch_log.update(
                    {f'{name}_loss' : value for name, value in losses.items()}
                )
                if criteria is not None:
                    batch_log.update(dict(
                        chain.from_iterable(
                            [
                                (name, output['value']),
                                (f'{name}_diagnostics', output)
                            ]
                            for name, output in performance.items()
                        )     
                    ))
                logger_train.log_batch(epoch_idx=i_epoch, batch_idx=i_batch, **batch_log)

        # Take weighted average of loss/accuracy across batches to get training values.
        if logger_train:
            batch_sizes = logger_train.get_all_entries(key='batch_size', level='batch', epoch_idx=i_epoch)
            train_mean_values = {
                key : logger_train.compute_weighted_sum(
                    key=key,
                    level='batch',
                    epoch_idx=i_epoch,
                    weights=batch_sizes/len(dataloader.dataset)
                )
                for key in batch_log.keys()
                if compute_mean_for is not None 
                and key in compute_mean_for
            }
            epoch_log.update(train_mean_values)
            logger_train.log_epoch(epoch_idx=i_epoch, **epoch_log)

        # Validate model on validation set after each epoch.
        with temporarily_eval(model):
            logger_val = eval.evaluate(model, **evaluation)

        if len(logger_val.epoch_logs) != 1:
            raise RuntimeError(
                "There should be only one \"epoch\" consisting of passing all " 
                f"validation data through model, but got {len(logger_val.epoch_logs)} epochs."
            )
        val_results = logger_val.get_entry(level='epoch', epoch_idx=0) 
        # if save_validation_logger: epoch_log['val_logger'] = validation_logger # The logged Logger object will not be easily JSON serializable
        # epoch_log.update(
        #     {f'val_{name}' : value for name, value in val_mean_values.items()}
        # )
        # Print validation results.
        
        # for name, value in val_results.items():
        #     print(f"val_{name}: {value} \n")
     
        # Optionally save model checkpoint.
        if metric_tracker:
            should_save, is_best = metric_tracker.should_save(
                val_results[metric_tracker.metric_name], i_epoch
            )
            if should_save:
                checkpoint = {
                    'epoch' : i_epoch,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dicts' : {
                        loss_term.name : loss_term.optimizer.state_dict()
                        for loss_term in loss_terms
                    },
                    'epoch_log' : epoch_log,
                    'best_epoch_so_far' : metric_tracker.best_epoch,
                    'best_value_so_far' : metric_tracker.best_value
                }
                checkpoint = recursion_utils.recursive(
                    checkpoint,
                    branch_conditionals=(
                        recursion_utils.dict_branch,
                        recursion_utils.list_branch,
                        recursion_utils.tuple_branch
                    ),
                    leaf_fns=(
                        tensor_utils.move_to_device('cpu'), 
                        tensor_utils.detach_tensor
                    )
                )
                
                metric_tracker.save(checkpoint, i_epoch, is_best=is_best)
       
        if early_stopping:
            early_stopping.update(val_results[early_stopping.metric_name])
            if early_stopping.should_stop(i_epoch):
                if early_stopping.verbose: 
                    early_stopping.print_to_console()
                return logger_train, logger_val, metric_tracker, early_stopping

        # If logger objects print, add demarcation between epochs.
        if logger_train.verbose_epoch or logger_val.verbose_epoch:
            print('----------------------------------------')

    return logger_train, logger_val, metric_tracker, early_stopping
    







































