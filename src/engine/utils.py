import torch

from general_utils import tensor as tensor_utils


def compute_accuracy(pred_labels, true_labels, masks):
    """ 
    """
    matches = torch.eq(pred_labels, true_labels) | ~masks
    seq_matches = matches.all(dim=1)

    num_correct = seq_matches.sum().item()
    num_seq = pred_labels.shape[0]

    correct_ind = torch.where(seq_matches)[0]
    incorrect_ind = torch.where(~seq_matches)[0]

    return {
        'value' : num_correct/num_seq, 
        'num_correct' : num_correct, 
        'num_seq' : num_seq, 
        'correct_ind' : correct_ind, 
        'incorrect_ind' : incorrect_ind
    }

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
    








