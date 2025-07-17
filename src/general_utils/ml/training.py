from pathlib import Path
import torch
from typing import Literal
import warnings

from .. import tensor as tensor_utils
from .config import RequiresGradConfig


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
        self.recent_vals[-1] = tensor_utils.tensor_to_cpu_python_scalar(x)

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


def set_requires_grad(
    model: torch.nn.Module, 
    cfg: RequiresGradConfig,
    # mode: Literal['inclusion', 'exclusion'],
    # requires_grad: bool,
    # verbose=True
):
    """ 
    """
    def set_value(named_params, networks, value):
        for network, patterns in networks.items():
            for pat in patterns:
                for param_name, param in named_params:
                    if param_name.startswith(network + '.') and pat in param_name:
                        param.requires_grad=value

    # Validate that all network names show up in model attributes.
    for network in cfg.networks.keys():
        if network not in model._modules:
            raise ValueError(
                f"cfg.networks key '{network}' is not a registered submodule of `model`."
            )

    named_params = list(model.named_parameters())

    if cfg.mode == 'inclusion':
        set_value(named_params, cfg.networks, cfg.requires_grad)
    elif cfg.mode == 'exclusion':
        for _, param in named_params:
            param.requires_grad = cfg.requires_grad
        set_value(named_params, cfg.networks, not(cfg.requires_grad))
    else:
        raise ValueError(
            f"Got unrecognized value {cfg.mode} for `mode`. Must be 'inclusion' or 'exclusion'."
        )
    if cfg.verbose:
        for param_name, param in named_params:
            if param.requires_grad:
                print(f"{param_name} is active.")
            else:
                print(f"{param_name} is frozen.")