import json
from pathlib import Path
import torch

from .. import tensor as tensor_utils


class Logger:
    """ 
    Utility class for logging per batch/epoch results during training or 
    testing. Can log an arbitrary number of items per batch/epoch.
    """
    def __init__(
            self, 
            log_dir: str, 
            log_name: str, 
            print_flush_epoch: bool = False, 
            print_flush_batch: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_name = log_name
        self.batch_logs = []
        self.epoch_logs = []
        self.print_flush_epoch = print_flush_epoch
        self.print_flush_batch = print_flush_batch
    
    def log_batch(self, *, epoch: int, batch: int, verbose=False, **kwargs):
        entry = {'epoch' : epoch, 'batch': batch}
        kwargs = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
        entry.update(kwargs)
        self.batch_logs.append(entry)
        if verbose:
            for key, value in kwargs.items():
                print(
                    f"{key} for epoch {epoch}, batch {batch}: {value}.", 
                    flush=self.print_flush_batch
                )

    def log_epoch(self, *, epoch: int, verbose=True, **kwargs):
        entry = {'epoch' : epoch}
        kwargs = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
        entry.update(kwargs)
        self.epoch_logs.append(entry)
        if verbose:
            for key, value in kwargs.items():
                print(
                    f"{key} for epoch {epoch}: {value}.", 
                    flush=self.print_flush_epoch
                )

    def get_logged_values(self, key: str, level: str):
        """ 
        Retrieve all logged values so far, at either the batch or epoch level.
        Will raise KeyError if any entries are missing the requested key.
        """
        if level not in ['batch', 'epoch']:
            raise ValueError(
                f"Unrecognized value {level} for `level`. Must be 'batch' or 'epoch'."
            )
        source = self.epoch_logs if level == 'epoch' else self.batch_logs
        values = [entry[key] for entry in source if key in entry]
        if len(values) < len(source):
            raise KeyError(
                f"key '{key}' missing from {len(source)-len(values)} entries."
            )
        return torch.tensor(values)
    
    def compute_weighted_sum(self, key: str, level: str, weights: torch.Tensor):
        values = self.get_logged_values(key=key, level=level)
        tensor_utils.validate_tensor(weights, 1)
        return torch.sum(values * weights)
    
    def save(self):
        batch_path = self.log_dir / f'{self.log_name}_batch.jsonl'
        epoch_path = self.log_dir / f'{self.log_name}_epoch.jsonl'

        with open(batch_path, 'w', newline='\n') as f:
            for entry in self.batch_logs:
                f.write(json.dumps(entry, sort_keys=True) + '\n')

        with open(epoch_path, 'w', newline='\n') as f:
            for entry in self.epoch_logs:
                f.write(json.dumps(entry, sort_keys=True) + '\n')

    def reset(self):
        self.batch_logs = []
        self.epoch_logs = []