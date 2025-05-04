from itertools import chain
from pathlib import Path
import torch
from typing import Callable
import warnings

from . import eval, utils
from ..utils import tensor as tensor_utils


class EarlyStopping:
    """ 
    """
    def __init__(
        self, 
        metric_name: str,
        patience: int,
        mode : str,
        min_epochs_before_stopping: int = 1,
        verbose=True,
        disabled=False
    ):
        """ 
        verbose is not used in any internal methods. Rather since this is an
        auxiliary class meant to function in a training environment, that
        environment can access the verbose attribute to know whether or not
        to print to console. 
        """
        if mode not in ['min', 'max']: 
            raise ValueError(
                f"Unrecognized value {mode} for `mode`. Must be one of ['min', 'max']."
            )
        self.metric_name = metric_name
        self.patience = patience
        self.mode = mode
        self.min_epochs_before_stopping = min_epochs_before_stopping
        self.verbose = verbose
        self.disabled = disabled

        self.recent_vals = torch.full((self.patience + 1,), torch.nan)
        self.counter = 0
        self.stopped_after_epoch = None

    def update(self, x):
        """ 
        """
        self.recent_vals = self.recent_vals.roll(-1)
        self.recent_vals[-1] = tensor_utils.to_python_scalar(x)

    def should_stop_early(self, epoch_idx):
        """ 
        """
        if self.disabled or epoch_idx + 1 < self.min_epochs_before_stopping: 
            return False

        diffs = torch.diff(self.recent_vals, n=1)
        if (
            (self.mode == 'min' and (diffs > 0).all())
            or (self.mode == 'max' and (diffs < 0).all())
        ):
            self.stopped_after_epoch = epoch_idx
            return True
        
        return False
        
    def print_to_console(self):
        if self.stopped_after_epoch is None:
            raise RuntimeError(
                "Attempting to print to console that early stopping condition " 
                "has been reached, but self.should_stop_early has not returned True yet."
            )
        print(
            f"Early stopping condition reached after epoch {self.stopped_after_epoch}.\n"
            f"Tracked value over last {self.patience+1} epochs: {self.recent_vals}.\n"
            f"Final changes prior to early stopping: {torch.diff(self.recent_vals, n=1)}."
        )
    
        
class MetricTracker:
    """ 
    """
    def __init__(self, metric_name, checkpoint_dir, mode: str, frequency: str ='best'):
        """ 
        mode : string
            ['min' | 'max']. 
        """
        if not mode in ['min', 'max']:
            raise ValueError(
                f"Unrecognized value {mode} for `mode`. Must be one of ['min', 'max']."
            )
        if not (
            frequency in ['best', 'always'] 
            or (isinstance(frequency, int) and frequency > 0)
        ):
            raise ValueError(
                f"Unrecognized value {frequency} for `frequency`. Must be "
                "either in ['best', 'always'], or a positive int."
            )
        
        self.metric_name = metric_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.mode = mode
        self.prev_best_value = None
        self.prev_best_epoch = None
        self.best_value = float('inf') if mode == 'min' else -float('inf') 
        self.best_epoch = None

    def get_checkpoint_path(self, epoch_idx, is_best=False, ext='.pt'):
        suffix = '_best' if is_best else ''
        return self.checkpoint_dir / f'{epoch_idx}{suffix}{ext}'

    def _update_if_is_best(self, value, epoch_idx):
        value = float(value)
        is_best = (
            value < self.best_value if self.mode == 'min' 
            else value > self.best_value 
        )
        if is_best:
            # Retain previous best epoch index for reference in `save` method.
            self.prev_best_value = self.best_value
            self.prev_best_epoch = self.best_epoch
            self.best_value = value
            self.best_epoch = epoch_idx
        return is_best
    
    def should_save(self, value, epoch_idx: int) -> bool:
        """ 
        Compare new value against current stored best value; if better,
        register this in the object's internal state. Then returns boolean
        2-tuple (a, b), where a is True iff saving condition satisfied, and b
        is True iff new value is improvement over current stored best value.
        """
        # Always register value and epoch index internally if is best so far.
        is_best = self._update_if_is_best(value, epoch_idx)

        # Return True if should save, else False.
        if self.frequency == 'always': 
            return (True, is_best)
        elif isinstance(self.frequency, int):
            return (epoch_idx % self.frequency == 0 or is_best, is_best)
        elif self.frequency == 'best':
            return (is_best, is_best)
        else:
            raise ValueError(
                f"Unrecognized value for `frequency`: {self.frequency}."
            )
        
    def save(self, checkpoint, epoch_idx, is_best=False):
        """ 
        Parameters
        ----------
        checkpoint : serializable object
        epoch_idx : int
        """
        # If current model is best, need to handle previously saved best models.
        if is_best:
            # Get filename of previous best model, if it exists.
            prev_best_path = (
                self.get_checkpoint_path(self.prev_best_epoch, is_best=True)
                if self.prev_best_epoch is not None else None
            )

            # Either overwrite previous best file or rename.
            if prev_best_path is not None:
                if (
                    self.frequency == 'best'
                    or (isinstance(self.frequency, int) and epoch_idx % self.frequency != 0)
                ):
                    prev_best_path.unlink() # Delete previous best file (overwrite)
                elif (
                    self.frequency == 'always'
                    or (isinstance(self.frequency, int) and epoch_idx % self.frequency == 0)
                ):
                    # Prepare new filename with '_best' suffix stripped.
                    plain_path = self.get_checkpoint_path(
                        self.prev_best_epoch, is_best=False
                    )
                    if plain_path.exists():
                        warnings.warn(
                            f"Renaming {prev_best_path} to {plain_path}, but "
                            f"{plain_path} already exists and will be overwritten!"
                        )
                    prev_best_path.rename(plain_path)
                else:
                    warnings.warn(
                        "Current model is best, and a previous best model has " 
                        f"been saved. However, self.frequency = {self.frequency} "
                        f"and epoch_idx = {epoch_idx} yielded an unrecognized "
                        "condition. The current best model will still be saved, "
                        "but there may be more than one file with suffix "
                        "'_best', which is not intended."
                    )
                
        torch.save(checkpoint, self.get_checkpoint_path(epoch_idx, is_best))

    def save_if_should_save(self, value, epoch_idx, checkpoint):
        """ 
        Convenience method to call `should_save` and then `save`, which could
        also be called manually in sequence.
        """
        should_save, is_best = self.should_save(value, epoch_idx)
        if should_save:
            self.save(checkpoint, epoch_idx, is_best=is_best)


# def _process_batch_train(model, h_0, batch, labels, lengths, masks, deterministic):
#     """ 
#     """
#     logits, rnn_output, _ = model(
#         batch, h_0=h_0, lengths=lengths, output_type='many_to_many'
#     )
#     _, pred_labels = model.to_token(logits, deterministic=deterministic)

#     # Masks and labels are shifted cyclically by one to align next token
#     # predictions with ground truth. This should be safe as response phase is
#     # always preceded in a sequence by other tokens.
#     accuracy = utils.compute_accuracy(
#         pred_labels, labels.roll(-1, dims=1), masks.roll(-1, dims=1)
#     )

#     return (logits, rnn_output), accuracy

def train(
    model, 
    train_loader, 
    loss_terms,
    evaluation,
    h_0=None,
    logger=None,
    criteria={'accuracy' : utils.compute_accuracy},
    compute_mean_for=None,
    save_validation_logger=True,
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
    if model.tokens.device != device:
        raise RuntimeError(
            f"model.tokens must be on user specified device {device} but is " 
            f"on {model.tokens.device}."
        )
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

    for i_epoch in range(num_epochs):
        if logger: epoch_log = {}

        for i_batch, (batch, labels, lengths, masks, seq_ind) in enumerate(train_loader):
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
                batch, h_0=h_0, lengths=lengths, output_type='many_to_many'
            )
            _, pred_labels = model.to_token(logits, deterministic=deterministic)

            # Compute losses.
            losses = {
                name : loss_term.compute_loss(logits, labels, model)
                for name, loss_term in loss_terms.items()
            }
            for loss_term in loss_terms.values(): loss_term.step()
            
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
            if logger: 
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
                logger.log_batch(epoch=i_epoch, batch=i_batch, **batch_log)

        # Take weighted average of loss/accuracy across batches to get training values.
        if logger:
            batch_sizes = logger.get_logged_values(key='batch_size', level='batch')
            train_mean_values = {
                key : logger.compute_weighted_sum(
                    key=key,
                    level='batch',
                    weights=batch_sizes/len(train_loader.dataset)
                )
                for key in batch_log.keys()
                if compute_mean_for is not None 
                and key in compute_mean_for
            }
            epoch_log.update(train_mean_values)

        # Validate model on validation set after each epoch.
        validation_logger, validation_mean_values = eval.evaluate(model, **evaluation)
        if save_validation_logger: epoch_log['val_logger'] = validation_logger
        epoch_log.update(
            {f'val_{name}' : value for name, value in validation_mean_values.items()}
        )
        
        if logger: logger.log_epoch(epoch=i_epoch, **epoch_log)

        # Optionally save model checkpoint.
        if metric_tracker:
            should_save, is_best = metric_tracker.should_save(
                epoch_log[metric_tracker.metric_name], i_epoch
            )
            if should_save:
                checkpoint = {
                    'epoch' : i_epoch,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dicts' : {
                        loss_term.name : loss_term.optimizer.state_dict()
                        for loss_term in loss_terms.values()
                    },
                    'epoch_log' : epoch_log,
                    'best_epoch_so_far' : metric_tracker.best_epoch,
                    'best_value_so_far' : metric_tracker.best_value
                }
                # TODO: Add logic to recursively check checkpoints and detach,
                # move to CPU, maybe move to numpy etc.
                metric_tracker.save(checkpoint, i_epoch, is_best=is_best)

        if early_stopping:
            early_stopping.update(epoch_log[early_stopping.metric_name])
            if early_stopping.should_stop_early(i_epoch):
                if early_stopping.verbose: 
                    early_stopping.print_to_console()
                return logger, metric_tracker, early_stopping

    return logger, metric_tracker, early_stopping, epoch_log
    
















# def train(
#     model: nn.Module,
#     dataloader: DataLoader,
#     loss_terms: list[LossTerm],
#     h_0: Optional[Callable],
#     ... # other args
# ):
#     ...


# from hydra.utils import instantiate
# from your_project.train import train

# cfg.train.h_0 = instantiate(cfg.train.h_0)
# cfg.train.loss_terms = [instantiate(term) for term in cfg.train.loss_terms]

# train(**cfg.train)





# # h_0:
# #   _target_: your_project.utils.make_h0_sampler
# #   distribution_cls: torch.distributions.Normal
# #   distribution_params:
# #     loc: 0.0
# #     scale: 1.0
# #   shape_fn:
# #     _target_: your_project.utils.make_shape_fn
# #     num_layers: 1
# #     hidden_size: 1000
# #   rsample: false
# #   device: cuda

# # from hydra.utils import instantiate

# # h_0 = instantiate(cfg.train.h_0)
# # train(..., h_0=h_0)


































# import inspect
# import torch
# from torch import nn
# import warnings

# from data import utils as data_utils


# def move_to_device(data, device):
#     """ 
#     Utility for moving tensors, including tensors in arbitrarily nested lists/tuple/dicts.
#     """
#     if isinstance(data, torch.Tensor): 
#         return data.to(device, non_blocking=True)
#     elif isinstance(data, (list, tuple)): 
#         return type(data)(move_to_device(elem, device) for elem in data)
#     elif isinstance(data, dict):
#         return {key : move_to_device(value, device) for key, value in data.items()}
#     else:
#         return data
    
# def early_stopping(
#     loss_by_epoch, 
#     delta, 
#     epsilon, 
#     criterion=lambda x : torch.mean(torch.diff(x, n=1))
# ):
#     """ 
#     """
#     if len(loss_by_epoch) < delta: 
#         return False
#     elif criterion(loss_by_epoch) > epsilon:
#         return True
#     else:
#         return False
    
# def shift_masks(masks):
#     """ 
#     """
#     if (masks[:, 0]).any().item():
#         invalid_rows = torch.where(masks[:, 0])[0]
#         raise ValueError(
#             "The first column of masks contains a True value in rows "
#             f"{invalid_rows.tolist()}. Since masks should be of the response "
#             "phase, there should always be sequence elements preceding the "
#             "masked part, and the first element in each mask should thus be False."
#         )
#     else:
#         return masks.roll(-1, dims=1)

# def compute_accuracy(pred_labels, true_labels, masks):
#     """ 
#     """
#     matches = torch.eq(pred_labels, true_labels) | ~masks
#     seq_matches = matches.all(dim=1)

#     num_correct = seq_matches.sum().item()
#     num_seq = pred_labels.shape[0]

#     correct_ind = torch.where(seq_matches)[0]
#     incorrect_ind = torch.where(~seq_matches)[0]

#     return num_correct/num_seq, num_correct, num_seq, correct_ind, incorrect_ind

# def train(
#     model, 
#     dataloader, 
#     optimizer, 
#     loss_fn=nn.CrossEntropyLoss(),
#     h_0=None,
#     sample_h_0=False,
#     early_stopping_params=None, # Dict with delta, epsilon, and criterion_fn keys, corresponding to parameters for early_stopping function
#     num_epochs=20, 
#     device='cuda',
#     pack=True,
#     deterministic=True,
# ):
#     """ 
#     Parameters
#     ----------
#     model (AutoRNN) : 
#     dataloader (DataLoader) : 
#     optimizer ()
#     loss_fn 
#     h_0
#     early_stopping_params
#     num_epochs
#     device
#     pack
#     deterministic (Bool): Controls whether tokens/labels based on logits are 
#         generated deterministically vai argmax or probabilistically via softmax 
#         and sampling.

#     Returns
#     -------
#     out 
#     """
#     model.to(device)
#     model.train()
    
#     # Ensure tokens device matches that specified by user.
#     if model.tokens.device != device:
#         raise RuntimeError(
#             f"model.tokens must be on user specified device {device} but is " 
#             f"on {model.tokens.device}."
#         )

#     # Store relevant items from training. 
#     out = {
#         'logits' : [], 
#         'hidden' : [], 
#         'lengths' : [],
#         'seq_ind' : [],
#         'loss' : torch.full((num_epochs,), torch.nan),
#         'accuracy' : torch.full((num_epochs,), torch.nan), 
#         'early_stop_epoch' : None
#     }

#     for i_epoch in range(num_epochs):
#         running_loss = 0
#         batch_counter = 0

#         epoch = {
#             'logits' : [],
#             'hidden' : [],
#             'lengths' : [],
#             'seq_ind' : [],
#             'accuracy' : []
#         }

#         # Check for valid input of h_0 and sample_h_0.
#         if sample_h_0:
#             if not isinstance(sample_h_0, dict): raise TypeError(
#                 "sample_h_0 should be either None, or a dict with keys 'distr' " 
#                 "and 'distr_params'."
#             )
#             if 'distr' not in sample_h_0: raise KeyError(
#                 "sample_h_0 shoule be a dict with keys 'distr' and 'distr_params'."
#             )
#             if not inspect.ismethoddescriptor(sample_h_0['distr']): raise ValueError(
#                 "Unrecognized value for sample_h_0['distr']. Should be a "
#                 "sampling method such as torch.Tensor.normal_, or None."
#             )
#             if h_0 is not None: warnings.warn(
#                 "h_0 was not passed in as None, but sample_h_0 was passed in "
#                 "as True, requesting random sampling of initial hidden states "
#                 "for each batch. This will be carried out, overriding the "
#                 "passed in value of h_0."
#             )

#         for batch, labels, lengths, masks, seq_ind in dataloader:
#             batch, labels, masks = batch.to(device), labels.to(device), masks.to(device)

#             optimizer.zero_grad()
        
#             if pack:
#                 lengths = lengths 
#             else:
#                 warnings.warn(
#                     "`pack` has been passed in as False. This means that all "
#                     "sequences should be of the same length."
#                 )
#                 lengths = None

#             # Optionally randomly sample initial hidden states.
#             if sample_h_0:
#                 h_0 = sample_h_0['distr'](
#                     torch.full(
#                         model.get_h_0_shape(batch_size=batch.shape[0]), 
#                         torch.nan
#                     ),
#                     **sample_h_0['distr_params'],
#                 ).to(device)

#             logits, rnn_output, _ = model(
#                 batch, h_0=h_0, lengths=lengths, output_type='many_to_many'
#             )

#             # Need to shift masks forward by one since prediction is of next token.
#             shifted_masks = shift_masks(masks)

#             # Compute accuracy (values already extracted as python float). Note 
#             # that, in addition to using shifted masks, the columns of labels 
#             # are shifted cyclically by one to align prediction of next tokens 
#             # with ground truth tokens. This should be safe to do since the 
#             # response phase is always preceded in a sequence by other tokens.
#             _, pred_labels = model.to_token(
#                 logits, deterministic=deterministic
#             )
#             epoch['accuracy'].append(
#                 compute_accuracy(
#                     pred_labels, 
#                     labels.roll(-1, dims=1), 
#                     shifted_masks
#                 )
#             )

#             # Compute loss. 
#             loss = loss_fn(logits[shifted_masks], labels[masks])
#             running_loss += loss.item()
#             batch_counter += 1 # TODO: should maybe count samples to get unbiased estimate of loss

#             loss.backward()
#             optimizer.step()

#             epoch['logits'].append(logits.detach().cpu())
#             epoch['hidden'].append(rnn_output.detach().cpu())
#             epoch['lengths'].append(lengths.detach().cpu())
#             epoch['seq_ind'].append(seq_ind.detach().cpu())

#         # Compute and print epoch loss and accuracy.
#         out['loss'][i_epoch] = running_loss / batch_counter
#         print(f"Mean batch loss for epoch {i_epoch}: {out['loss'][i_epoch]}.")
#         out['accuracy'][i_epoch] = torch.mean(
#             torch.tensor([batch_acc[0] for batch_acc in epoch['accuracy']])
#         )
#         print(f"Mean batch accuracy for epoch {i_epoch}: {out['accuracy'][i_epoch]}.")

#         # Move items to CPU to save GPU memory and store.
#         for key, value in epoch.items(): 
#             if isinstance(out[key], list): out[key].append(value) 

#         # Optional early stopping.
#         if early_stopping_params is not None and early_stopping(
#             out['loss'][~torch.isnan(out['loss'])],
#             **early_stopping_params,
#         ):
#             loss_not_nan = out['loss'][~torch.isnan(out['loss'])]
#             print(
#                 f"Early stopping condition reached after epoch {i_epoch}. "
#                 f"Loss over last {early_stopping_params['delta']} epochs: "
#                 f"{loss_not_nan[-early_stopping_params['delta']:]}."
#                 "\nFinal loss changes prior to early stopping: "
#                 f"{torch.diff(loss_not_nan[-early_stopping_params['delta']:])}."
#             )
#             out['early_stop_epoch'] = i_epoch

#             return out
        
#     return out

# def test(
#     model, 
#     dataloader, 
#     switch_label,
#     loss_fn, 
#     criteria=None,
#     h_0=None,
#     move_results_to_cpu=True,
#     deterministic=True,
#     device='cuda',
# ):
#     """ 
#     """
#     model.to(device)
#     model.eval()
    
#     out = {
#         'mean_loss' : None,
#         'loss' : [],
#         'performance' : [],
#         'generated' : [],
#         'on_input' : [],
#         'joined' : [],
#         'labels' : [],
#         'masks' : [],
#         'seq_ind' : []
#     }

#     loss = 0
#     num_batches = len(dataloader)
#     batch_sizes = torch.full((num_batches,), torch.nan)

#     with torch.no_grad():
#         for i_batch, (batch_full_seq, labels, lengths, masks, seq_ind) in enumerate(dataloader):
#             # Get batch of demo phase portions of sequences by indexing up to 
#             # latest occurence, across seqeunces, of switch token.
#             _, switch_ind = torch.where(labels == switch_label)
#             max_switch_idx = torch.max(switch_ind) 
#             batch_demo_phase = batch_full_seq[:, :max_switch_idx+1, :]
            
#             # Lengths must be on CPU, thus move things to GPU after this step.
#             demo_lengths = switch_ind + 1 # Up to and including switch token
#             resp_lengths = lengths - switch_ind - 1 # From first count token to EOS, inclusive

#             batch_demo_phase = batch_demo_phase.to(device)
#             labels = labels.to(device)
#             masks = masks.to(device)
            
#             # if max_resp_len is None: max_resp_len = torch.max(resp_lengths)

#             generated, on_input, joined = model.generate(
#                 batch_demo_phase, 
#                 input_lengths=demo_lengths,
#                 h_0=h_0,  
#                 resp_lengths=resp_lengths,
#                 deterministic=deterministic,
#             )     

#             # Now implemented inside generate method:
#             # # Mask to recover valid portion of generated logit tensor. 
#             # batch_size = batch_demo_phase.shape[0]
#             # generated['resp_masks'] = torch.full(
#             #     (batch_size, max_resp_len),
#             #     fill_value=False,
#             #     device=device
#             # )
#             # for i_seq in range(batch_size):
#             #     generated['resp_masks'][i_seq, :resp_lengths[i_seq]] = True

#             # Get reshaped views for computation of loss.
#             gen_logits_reshaped = torch.reshape(
#                 generated['logits'],
#                 (-1, generated['logits'].shape[-1])
#             )
#             gen_resp_masks_reshaped = torch.reshape(generated['resp_masks'], (-1,))
            
#             loss = loss_fn(
#                 gen_logits_reshaped[gen_resp_masks_reshaped, :], 
#                 labels[masks].to(torch.int64)
#             )
#             if criteria is not None:
#                 # Need to get response portions of ground truth but left-aligned
#                 # at first count token after switch token.
#                 resp_start_col_ind = switch_ind + 1
#                 resp_stop_col_ind = switch_ind + 1 + torch.max(resp_lengths)
#                 resp_col_ind = torch.stack([
#                     torch.arange(start, stop, device=device) 
#                     for start, stop in zip(resp_start_col_ind, resp_stop_col_ind)    
#                 ])
#                 true_resp_labels_aligned = torch.gather(
#                     labels, dim=1, index=resp_col_ind
#                 )
#                 # true_resp_masks_aligned = torch.gather(
#                 #     masks, dim=1, index=resp_col_ind
#                 # )

#                 # Output should be detached.
#                 performance = [
#                     criterion(
#                         generated['labels'],  
#                         true_resp_labels_aligned,
#                         generated['resp_masks']
#                     ) 
#                     for criterion in criteria
#                 ]
#             else:
#                 performance = None

#             batch_sizes[i_batch] = batch_full_seq.shape[0]

#             out['loss'].append(loss)
#             out['performance'].append(performance)
#             out['generated'].append(generated) 
#             out['on_input'].append(on_input)
#             out['joined'].append(joined)
#             out['labels'].append(labels)
#             out['masks'].append(masks)
#             out['seq_ind'].append(seq_ind)
        
#     out['loss'] = torch.tensor(out['loss'])
#     out['mean_loss'] = torch.sum(
#         (batch_sizes/len(dataloader.dataset)) * out['loss']
#     )

#     if move_results_to_cpu:
#         for key, value in out.items(): out[key] = move_to_device(value, 'cpu')

#     return out
 
# # def join_demo_gen_resp(demo, demo_lengths, resp, resp_lengths):
# #     """ 
# #     Utility to concatenate itmes (e.g. logits/hidden states but most the latter) 
# #     from demonstration phase input and generated output, accounting
# #     for timesteps that may appear in both (e.g. at the switch index). 

# #     demo and resp should be of shape (b, n1, h) and (b, n2, h), where b and h 
# #     are the batch and hidden size, respectively, and n1 and n2 are the padded 
# #     lengths of the demonstration phase input and generated response, repsectively.
# #     """
# #     # Validate inputs.
# #     data_utils.validate_tensor(demo, 3)
# #     data_utils.validate_tensor(resp, 3)
# #     if not(demo.shape[0] == resp.shape[0] and demo.shape[2] == resp.shape[2]):
# #         raise ValueError(
# #             "The sizes of the first and third dimensions of `demo` and `resp` must match."
# #         )
# #     data_utils.validate_tensor(demo_lengths, 1)
# #     data_utils.validate_tensor(resp_lengths, 1)
# #     if len(demo_lengths) != len(resp_lengths):
# #         raise ValueError(
# #             "The lengths of `demo_lengths` and `resp_lengths` must match, but " 
# #             f"got {len(demo_lengths)} and {len(resp_lengths)}, respectively."
# #         )

# #     # Preallocate.
# #     batch_size, _, hidden_size = demo.shape
# #     max_input_len = torch.max(demo_lengths)
# #     max_resp_len = torch.max(resp_lengths)
# #     joined_padded = torch.full(
# #         (batch_size, max_input_len+max_resp_len, hidden_size), fill_value=0.0
# #     )
# #     joined_lengths = torch.full((batch_size,), 0, dtype=torch.int64)

# #     # Benchmarking suggests using masks and list comprehension is slightly slower.
# #     for i_seq in range(batch_size):
# #         # Get new set of lengths that identifies joined sequences as valid 
# #         # portions of padded output.
# #         joined_lengths[i_seq] = demo_lengths[i_seq] + resp_lengths[i_seq]

# #         # Join each sequence.
# #         joined_padded[i_seq, :joined_lengths[i_seq], :] = torch.cat(
# #             (
# #                 demo[i_seq, :demo_lengths[i_seq], :], 
# #                 resp[i_seq, :resp_lengths[i_seq], :]
# #             ),
# #             dim=0
# #         )
        
# #     return joined_padded, joined_lengths



