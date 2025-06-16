from dataclasses import asdict, is_dataclass

import torch


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
    samples from this distribution.

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

# def recursive(x, func):
#     """ 
#     """
#     if isinstance(x, torch.Tensor):
#         return func(x)
#     elif isinstance(x, dict):
#         return {k : recursive(v, func) for k, v in x.items()}
#     elif isinstance(x, (list, tuple)):
#         return type(x)(recursive(v, func) for v in x)
#     else:
#         return x

# def move_to_device(x, device, detach=False):
#     """ 
#     Utility for moving tensors, including tensors in arbitrarily nested lists/tuple/dicts.
#     """
#     return recursive(
#         x, 
#         lambda x: (x.detach() if detach else x).to(device, non_blocking=True)
#     )
    
# def tensor_to_cpu_python_scalar(x):
#     """ 
#     Extracts any one-element tensors (including those arbitrarily nested in
#     dicts/lists/tuples) as Python floats and moves them to the CPU. Other
#     objects are returned unchanged.
#     """
#     def func(x):
#         if x.numel() == 1: 
#             return x.cpu().item()
#         else:
#             raise ValueError(
#                 "Valid log entries may have only 1 element but got " 
#                 f"{x.numel()} elements with shape {x.shape}."
#             )
#     return recursive(x, func)
    
# def detach(x, to_cpu=True):
#     """
#     Recursively detach, optionally move to CPU.
#     """
#     return recursive(x, lambda x: x.detach().to('cpu' if to_cpu else x.device))
    
# def to_numpy(x):
#     """ 
#     """
#     return recursive(x, lambda x: x.numpy())
    
# def to_numpy(x):
#     """ 
#     """
#     if isinstance(x, torch.Tensor):
#         return x.numpy()
#     elif isinstance(x, dict):
#         return {k : to_numpy(v) for k, v in x.items()}
#     elif isinstance(x, (list, tuple)):
#         return type(x)(to_numpy(v) for v in x)
#     else:
#         return x  

# def detach(x, to_cpu=True):
#     """
#     Recursively detach, optionally move to CPU.
#     """
#     device = 'cpu' if to_cpu else x.device
#     if isinstance(x, torch.Tensor):
#         return x.detach().to(device)
#     elif isinstance(x, dict):
#         return {k : detach(v) for k, v in x.items()}
#     elif isinstance(x, (list, tuple)):
#         return type(x)(detach(v) for v in x)
#     else:
#         return x

# def tensor_to_cpu_python_scalar(x):
#     """ 
#     Extracts any one-element tensors (including those arbitrarily nested in
#     dicts/lists/tuples) as Python floats and moves them to the CPU. Other
#     objects are returned unchanged.
#     """
#     if isinstance(x, torch.Tensor):
#         if x.numel() == 1: 
#             return x.cpu().item()
#         else:
#             raise ValueError(
#                 "Valid log entries may have only 1 element but got " 
#                 f"{x.numel()} elements with shape {x.shape}."
#             )
#     elif isinstance(x, dict):
#         return {k : tensor_to_cpu_python_scalar(v) for k, v in x.items()}
#     elif isinstance(x, (list, tuple)):
#         return type(x)(tensor_to_cpu_python_scalar(v) for v in x)
#     else:
#         return x

# def move_to_device(data, device, detach=False):
#     """ 
#     Utility for moving tensors, including tensors in arbitrarily nested lists/tuple/dicts.
#     """
#     if isinstance(data, torch.Tensor): 
#         data = data.detach() if detach else data
#         return data.to(device, non_blocking=True)
#     elif isinstance(data, (list, tuple)): 
#         return type(data)(move_to_device(elem, device) for elem in data)
#     elif isinstance(data, dict):
#         return {key : move_to_device(value, device) for key, value in data.items()}
#     else:
#         return data

def recursive(x, *functions):
    """ 
    functions can be defined functions or lambda functions.
    """
    if isinstance(x, dict):
        return {k : recursive(v, *functions) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive(v, *functions) for v in x)
    else:
        for func in functions: 
            x = func(x)
        return x
    
def move_to_device(device):
    """ 
    Utility for configuring lambda function for moving tensor to specified
    device. Intended to be used with `recursive`.
    """
    return lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    
def tensor_to_cpu_python_scalar(x):
    """ 
    Extracts a one-element tensor as a Python scalar (float, int, etc.). Other 
    objects are returned unchanged.
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




