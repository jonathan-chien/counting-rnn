from itertools import chain

import torch

from . import eval, utils
from general_utils import recursion as recursion_utils
from general_utils import tensor as tensor_utils


# class EarlyStopping:
#     """ 
#     """
#     def __init__(
#         self, 
#         metric_name: str,
#         patience: int,
#         mode : str,
#         min_epochs_before_stopping: int = 1,
#         verbose=True,
#         disabled=False
#     ):
#         """ 
#         verbose is not used in any internal methods. Rather since this is an
#         auxiliary class meant to function in a training environment, that
#         environment can access the verbose attribute to know whether or not
#         to print to console. 
#         """
#         if mode not in ['min', 'max']: 
#             raise ValueError(
#                 f"Unrecognized value {mode} for `mode`. Must be one of ['min', 'max']."
#             )
#         self.metric_name = metric_name
#         self.patience = patience
#         self.mode = mode
#         self.min_epochs_before_stopping = min_epochs_before_stopping
#         self.verbose = verbose
#         self.disabled = disabled

#         self.recent_vals = torch.full((self.patience + 1,), torch.nan)
#         self.counter = 0
#         self.stopped_after_epoch = None

#     def update(self, x):
#         """ 
#         """
#         self.recent_vals = self.recent_vals.roll(-1)
#         self.recent_vals[-1] = tensor_utils.tensor_to_cpu_python_scalar(x)

#     def should_stop_early(self, epoch_idx):
#         """ 
#         """
#         if self.disabled or epoch_idx + 1 < self.min_epochs_before_stopping: 
#             return False

#         diffs = torch.diff(self.recent_vals, n=1)
#         if (
#             (self.mode == 'min' and (diffs > 0).all())
#             or (self.mode == 'max' and (diffs < 0).all())
#         ):
#             self.stopped_after_epoch = epoch_idx
#             return True
        
#         return False
        
#     def print_to_console(self):
#         if self.stopped_after_epoch is None:
#             raise RuntimeError(
#                 "Attempting to print to console that early stopping condition " 
#                 "has been reached, but self.should_stop_early has not returned True yet."
#             )
#         print(
#             f"Early stopping condition reached after epoch {self.stopped_after_epoch}.\n"
#             f"Tracked value over last {self.patience+1} epochs: {self.recent_vals}.\n"
#             f"Final changes prior to early stopping: {torch.diff(self.recent_vals, n=1)}."
#         )
    
        
# class MetricTracker:
#     """ 
#     """
#     def __init__(self, metric_name, checkpoint_dir, mode: str, frequency: str ='best'):
#         """ 
#         mode : string
#             ['min' | 'max']. 
#         """
#         if not mode in ['min', 'max']:
#             raise ValueError(
#                 f"Unrecognized value {mode} for `mode`. Must be one of ['min', 'max']."
#             )
#         if not (
#             frequency in ['best', 'always'] 
#             or (isinstance(frequency, int) and frequency > 0)
#         ):
#             raise ValueError(
#                 f"Unrecognized value {frequency} for `frequency`. Must be "
#                 "either in ['best', 'always'], or a positive int."
#             )
        
#         self.metric_name = metric_name
#         self.checkpoint_dir = Path(checkpoint_dir)
#         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
#         self.frequency = frequency
#         self.mode = mode
#         self.prev_best_value = None
#         self.prev_best_epoch = None
#         self.best_value = float('inf') if mode == 'min' else -float('inf') 
#         self.best_epoch = None

#     def get_checkpoint_path(self, epoch_idx, is_best=False, ext='.pt'):
#         suffix = '_best' if is_best else ''
#         return self.checkpoint_dir / f'{epoch_idx}{suffix}{ext}'

#     def _update_if_is_best(self, value, epoch_idx):
#         value = float(value)
#         is_best = (
#             value < self.best_value if self.mode == 'min' 
#             else value > self.best_value 
#         )
#         if is_best:
#             # Retain previous best epoch index for reference in `save` method.
#             self.prev_best_value = self.best_value
#             self.prev_best_epoch = self.best_epoch
#             self.best_value = value
#             self.best_epoch = epoch_idx
#         return is_best
    
#     def should_save(self, value, epoch_idx: int) -> bool:
#         """ 
#         Compare new value against current stored best value; if better,
#         register this in the object's internal state. Then returns boolean
#         2-tuple (a, b), where a is True iff saving condition satisfied, and b
#         is True iff new value is improvement over current stored best value.
#         """
#         # Always register value and epoch index internally if is best so far.
#         is_best = self._update_if_is_best(value, epoch_idx)

#         # Return True if should save, else False.
#         if self.frequency == 'always': 
#             return (True, is_best)
#         elif isinstance(self.frequency, int):
#             return (epoch_idx % self.frequency == 0 or is_best, is_best)
#         elif self.frequency == 'best':
#             return (is_best, is_best)
#         else:
#             raise ValueError(
#                 f"Unrecognized value for `frequency`: {self.frequency}."
#             )
        
#     def save(self, checkpoint, epoch_idx, is_best=False):
#         """ 
#         Parameters
#         ----------
#         checkpoint : serializable object
#         epoch_idx : int
#         """
#         # If current model is best, need to handle previously saved best models.
#         if is_best:
#             # Get filename of previous best model, if it exists.
#             prev_best_path = (
#                 self.get_checkpoint_path(self.prev_best_epoch, is_best=True)
#                 if self.prev_best_epoch is not None else None
#             )

#             # Either overwrite previous best file or rename.
#             if prev_best_path is not None:
#                 if (
#                     self.frequency == 'best'
#                     or (isinstance(self.frequency, int) and epoch_idx % self.frequency != 0)
#                 ):
#                     prev_best_path.unlink() # Delete previous best file (overwrite)
#                 elif (
#                     self.frequency == 'always'
#                     or (isinstance(self.frequency, int) and epoch_idx % self.frequency == 0)
#                 ):
#                     # Prepare new filename with '_best' suffix stripped.
#                     plain_path = self.get_checkpoint_path(
#                         self.prev_best_epoch, is_best=False
#                     )
#                     if plain_path.exists():
#                         warnings.warn(
#                             f"Renaming {prev_best_path} to {plain_path}, but "
#                             f"{plain_path} already exists and will be overwritten!"
#                         )
#                     prev_best_path.rename(plain_path)
#                 else:
#                     warnings.warn(
#                         "Current model is best, and a previous best model has " 
#                         f"been saved. However, self.frequency = {self.frequency} "
#                         f"and epoch_idx = {epoch_idx} yielded an unrecognized "
#                         "condition. The current best model will still be saved, "
#                         "but there may be more than one file with suffix "
#                         "'_best', which is not intended."
#                     )
                
#         torch.save(checkpoint, self.get_checkpoint_path(epoch_idx, is_best))

#     def save_if_should_save(self, value, epoch_idx, checkpoint):
#         """ 
#         Convenience method to call `should_save` and then `save`, which could
#         also be called manually in sequence.
#         """
#         should_save, is_best = self.should_save(value, epoch_idx)
#         if should_save:
#             self.save(checkpoint, epoch_idx, is_best=is_best)


# def set_requires_grad(
#     model: torch.nn.Module, 
#     cfg: RequiresGradConfig,
#     mode: Literal['inclusion', 'exclusion'],
#     requires_grad: bool,
#     verbose=True
# ):
#     """ 
#     """
#     def set_value(named_params, networks, value):
#         for network, patterns in networks.items():
#             for pat in patterns:
#                 for param_name, param in named_params:
#                     if param_name.startswith(network + '.') and pat in param_name:
#                         param.requires_grad=value

#     # Validate that all network names show up in model attributes.
#     for network in cfg.networks.keys():
#         if network not in model._modules:
#             raise ValueError(
#                 f"cfg.networks key '{network}' is not a registered submodule of `model`."
#             )

#     named_params = list(model.named_parameters())

#     if mode == 'inclusion':
#         set_value(named_params, cfg.networks, requires_grad)
#     elif mode == 'exclusion':
#         for _, param in named_params:
#             param.requires_grad = requires_grad
#         set_value(named_params, cfg.networks, not(requires_grad))
#     else:
#         raise ValueError(
#             f"Got unrecognized value {mode} for `mode`. Must be 'inclusion' or 'exclusion'."
#         )
#     if verbose:
#         for param_name, param in named_params:
#             if param.requires_grad:
#                 print(f"{param_name} is active.")
#             else:
#                 print(f"{param_name} is frozen.")

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
            shifted_masks = torch.roll(masks, shifts=-1, dims=0)

            # Compute losses.
            losses = {
                loss_term.name : loss_term.compute_loss(
                    logits[shifted_masks], labels[masks], model
                )
                for loss_term in loss_terms
            }
            for loss_term in loss_terms: loss_term.step()
            
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
            batch_sizes = logger_train.get_logged_values(key='batch_size', level='batch')
            train_mean_values = {
                key : logger_train.compute_weighted_sum(
                    key=key,
                    level='batch',
                    weights=batch_sizes/len(dataloader.dataset)
                )
                for key in batch_log.keys()
                if compute_mean_for is not None 
                and key in compute_mean_for
            }
            epoch_log.update(train_mean_values)
            logger_train.log_epoch(epoch_idx=i_epoch, **epoch_log)

        # Validate model on validation set after each epoch.
        logger_val = eval.evaluate(model, **evaluation)
        if len(logger_val.epoch_logs) != 1:
            raise RuntimeError(
                "There should be only one \"epoch\" consisting of passing all " 
                f"validation data through model, but got {len(logger_val.epoch_logs)} epochs."
            )
        val_results = logger_val.get_epoch(epoch_idx=0) 
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
            if early_stopping.should_stop_early(i_epoch):
                if early_stopping.verbose: 
                    early_stopping.print_to_console()
                return logger_train, logger_val, metric_tracker, early_stopping

        # If logger objects print, add demarcation between epochs.
        if logger_train.verbose_epoch or logger_val.verbose_epoch:
            print('----------------------------------------')

    return logger_train, logger_val, metric_tracker, early_stopping
    







































