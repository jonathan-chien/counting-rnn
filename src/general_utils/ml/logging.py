import copy
import json
from pathlib import Path
import torch
from typing import Sequence, Union

from ..config import TensorConfig
from .. import recursion as recursion_utils
from .. import serialization as serialization_utils
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
            verbose_batch: bool = False,
            verbose_epoch: bool = True,
            print_flush_epoch: bool = False, 
            print_flush_batch: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_name = log_name
        self.batch_logs = {}
        self.epoch_logs = {}
        self.verbose_batch = verbose_batch
        self.verbose_epoch = verbose_epoch
        self.print_flush_epoch = print_flush_epoch
        self.print_flush_batch = print_flush_batch
    
    # def log_batch(self, *, epoch: int, batch: int, verbose=False, **kwargs):
    #     entry = {'epoch_idx': epoch, 'batch_idx': batch}
    #     kwargs = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
    #     entry.update(kwargs)
    #     self.batch_logs.append(entry)
    #     if verbose:
    #         for key, value in kwargs.items():
    #             print(
    #                 f"{key} for epoch {epoch}, batch {batch}: {value}.", 
    #                 flush=self.print_flush_batch
    #             )

    def log_batch(self, *, epoch_idx: int, batch_idx: int, suppress_print=False, **kwargs):
        entry = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
        self.batch_logs[(epoch_idx, batch_idx)] = entry
        # self.batch_logs.append(entry)
        if self.verbose_batch and not suppress_print:
            for key, value in entry.items():
                print(
                    f"{self.log_name} {key} for epoch {epoch_idx}, batch {batch_idx}: {value}.", 
                    flush=self.print_flush_batch
                )

    # def log_epoch(self, *, epoch: int, verbose=True, **kwargs):
    #     entry = {'epoch_idx' : epoch}
    #     kwargs = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
    #     entry.update(kwargs)
    #     self.epoch_logs.append(entry)
    #     if verbose:
    #         for key, value in kwargs.items():
    #             print(
    #                 f"{key} for epoch {epoch}: {value}.", 
    #                 flush=self.print_flush_epoch
    #             )
    def log_epoch(self, *, epoch_idx: int, suppress_print=False, **kwargs):
        entry = {key : tensor_utils.tensor_to_cpu_python_scalar(val) for key, val in kwargs.items()}
        self.epoch_logs[epoch_idx] = entry
        if self.verbose_epoch and not suppress_print:
            for key, value in entry.items():
                print(
                    f"{self.log_name} {key} for epoch {epoch_idx}: {value}.", 
                    flush=self.print_flush_epoch
                )

    def get_batch(self, epoch_idx: int, batch_idx: int):
        """ 
        """
        try:
            return copy.deepcopy(self.batch_logs[(epoch_idx, batch_idx)])
        except KeyError:
            raise IndexError(
                f"No entry found for epoch {epoch_idx}, batch {batch_idx}."
            )

    def get_epoch(self, epoch_idx: int):
        """ 
        """
        try:
            return copy.deepcopy(self.epoch_logs[epoch_idx])
        except KeyError:
            raise IndexError(
                f"No entry found for epoch {epoch_idx}."
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
        try:
            values = [entry[key] for entry in source.values()]
        except KeyError:
            raise KeyError(
                f"The key '{key}' is missing from one or more {level} entries."
            )
        return torch.tensor(values)
    
    def compute_weighted_sum(self, key: str, level: str, weights: torch.Tensor):
        values = self.get_logged_values(key=key, level=level)
        tensor_utils.validate_tensor(weights, 1)
        return torch.sum(values * weights)
    
    def convert_to_serializable_format(self, target: Union[str, Sequence[str]]):
        """ 
        JSON serializable items plus dataclasses can be saved.
        """
        def _convert(log):
            # Convert tensors to TensorConfigs.
            log = tensor_utils.recursive_tensor_to_tensor_config(log)

            # Converts all dataclasses (including TensorConfig objects that may
            # have been generated above and their TensorArgsConfig attributes)
            # to tagged dicts.
            log = serialization_utils.recursive_dataclass_to_tagged_dict(log)

            return log
        
        if isinstance(target, str):
            target = [target]
        
        for log_name in target:
            if log_name not in ('batch_logs', 'epoch_logs'):
                raise ValueError(
                    "Unrecognized value for `target`. Must be one of " 
                    f"['batch_logs', 'epoch_logs'] but got '{log_name}'."
                ) 
            setattr(self, log_name, _convert(getattr(self, log_name)))
        

    def recover_from_serializable_format(self, target: Union[str, Sequence[str]]):
        """ 
        """
        def _recover(log):
            log = serialization_utils.recursive_tagged_dict_to_dataclass(log)
            log = serialization_utils.recursive_recover(log)
            return log

        if isinstance(target, str):
            target = [target]
        
        for log_name in target:
            if log_name not in ('batch_logs', 'epoch_logs'):
                raise ValueError(
                    "Unrecognized value for `target`. Must be one of " 
                    f"['batch_logs', 'epoch_logs'] but got '{log_name}'."
                ) 
            setattr(self, log_name, _recover(getattr(self, log_name)))
            
    
    def save(self):
        """ 
        """
        batch_path = self.log_dir / f'{self.log_name}_batch_log.jsonl'
        epoch_path = self.log_dir / f'{self.log_name}_epoch_log.jsonl'

        with open(batch_path, 'w', newline='\n') as f:
            for (epoch_idx, batch_idx), entry in self.batch_logs.items():
                entry = {**entry, 'epoch_idx': epoch_idx, 'batch_idx': batch_idx}
                f.write(json.dumps(entry, sort_keys=True) + '\n')

        with open(epoch_path, 'w', newline='\n') as f:
            for epoch_idx, entry in self.epoch_logs.items():
                entry = {**entry, 'epoch_idx': epoch_idx}
                f.write(json.dumps(entry, sort_keys=True) + '\n')

    def reset(self):
        self.batch_logs = {}
        self.epoch_logs = {}