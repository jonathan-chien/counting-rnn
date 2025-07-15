import torch

from .config import TensorConfig
from . import recursion as recursion_utils


class DistributionSampler:

    def __init__(self, distr, device=None):
        self.distr = distr
        self.device = device

    def __call__(self, sample_shape):
        if hasattr(self.distr, 'rsample'):
            sample = self.distr.rsample(sample_shape=sample_shape)
        else:
            sample = self.distr.sample(sample_shape=sample_shape)
        return sample.to(self.device) if self.device is not None else sample


def validate_tensor(tensor, dim, dtype=None):
    """ 
    Ensure that input is a torch tensor of specified dimension.
    """
    if not isinstance(tensor, torch.Tensor): 
        raise TypeError(f"Must be torch.Tensor but got {type(tensor)}.") 
    if tensor.dim() != dim: 
        raise ValueError(f"Must be {dim}d torch.Tensor.")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"Expected data type {dtype}, but got {tensor.dtype}.")
    
def make_sampler(distr, device=None):
    """ 
    Given a torch.distributions.Distribution subclass object, configures and
    returns a function capable of acceptin a sample size argument and returning
    samples from this distribution. Note that the returned sampler function is
    not pickleable. For a pickable solution, use the DistributionSampler class.

    Parameters
    ----------
    distr : torch.distributions.Distribution subclass instance
        Instance of torch.distributions.Distribution subclass, e.g. an instance
        of the torch.distributions.Normal class.

    Returns
    -------
    sampler : function
        Function accepting a sample shape tuple and returning sample tensor
        of that shape, each element of which is drawn from the specified 
        distribution.
    """
    def sampler(sample_shape):
        if hasattr(distr, 'rsample'):
            sample = distr.rsample(sample_shape=sample_shape)
        else:
            sample = distr.sample(sample_shape=sample_shape)
        return sample.to(device) if device is not None else sample
    
    return sampler

# def recursive(x, *functions):
#     """ 
#     functions can be defined functions or lambda functions.
#     """
#     if isinstance(x, dict):
#         return {k : recursive(v, *functions) for k, v in x.items()}
#     elif isinstance(x, (list, tuple)):
#         return type(x)(recursive(v, *functions) for v in x)
#     else:
#         for func in functions: 
#             x = func(x)
#         return x
    
def move_to_device(device):
    """ 
    Utility for configuring lambda function for moving tensor to specified
    device. Intended to be used with `recursive`.
    """
    return lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    
def tensor_to_cpu_python_scalar(x):
    """ 
    Extracts a one-element tensor as a Python scalar (float, int, etc.) on the 
    CPU. Other objects are returned unchanged.
    """
    return x.cpu().item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x

def detach_tensor(x):
    """
    Detach tensor.
    """
    return x.detach() if isinstance(x, torch.Tensor) else x
    
def tensor_to_numpy(x):
    """ 
    """
    return x.numpy() if isinstance(x, torch.Tensor) else x

def recursive_tensor_to_tensor_config(x):
    """ 
    Can be used to convert tensors nested in other structures for JSON serialization.
    """
    return recursion_utils.recursive(
        x,
        branch_conditionals=(
            recursion_utils.dict_branch,
            recursion_utils.list_branch,
            recursion_utils.tuple_branch
        ),
        leaf_fns=(
            lambda x: (
                TensorConfig.from_tensor(x) 
                if isinstance(x, torch.Tensor) else x
            ),
        )
    )





