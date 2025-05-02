import json
from pathlib import Path
import torch

from ..utils import tensor as tensor_utils


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
        kwargs = {key : tensor_utils.to_python_scalar(val) for key, val in kwargs}
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
        kwargs = {key : tensor_utils.to_python_scalar(val) for key, val in kwargs}
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

    # @staticmethod
    # def _detach(x):
    #     """ 
    #     Utility that accepts a dict and returns a new dict, where any 
    #     torch.Tensor values are expected to have only one value and will be
    #     extracted as a detached scalar and moved to the CPU.
    #     """
    #     y = {}
    #     for key, value in x.items():
    #         if isinstance(value, torch.Tensor):
    #             if value.numel() == 1: 
    #                 y[key] = value.cpu().item()
    #             else:
    #                 raise ValueError(
    #                     f"Valid log entries may have only 1 element but got {value.numel()}."
    #                 )
    #         else:
    #             y[key] = value



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

def calculate_prop_correct(pred_labels, true_labels, masks):
    """ 
    """
    matches = torch.eq(pred_labels, true_labels) | ~masks
    seq_matches = matches.all(dim=1)

    num_correct = seq_matches.sum().item()
    num_seq = pred_labels.shape[0]

    correct_ind = torch.where(seq_matches)[0]
    incorrect_ind = torch.where(~seq_matches)[0]

    return num_correct/num_seq, num_correct, num_seq, correct_ind, incorrect_ind

def validate_h_0_config(h_0, allow_grad=False):
    """ 
    """
    if h_0 is None or callable(h_0): 
        return h_0
    elif isinstance(h_0, torch.Tensor):
        tensor_utils.validate_tensor(h_0, dim=1)
        if h_0.requires_grad and not allow_grad:
            raise RuntimeError(
                "h_0.required_grad = True, but `allow_grad` is False."
            )
        return h_0
    else:
        raise ValueError(
            f"Unsupported type {type(h_0)} for `h_0`. Must be None, callable, " 
            "or torch.Tensor."
        )
    








