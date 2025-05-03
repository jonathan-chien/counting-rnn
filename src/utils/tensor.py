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

def move_to_device(data, device, detach=False):
    """ 
    Utility for moving tensors, including tensors in arbitrarily nested lists/tuple/dicts.
    """
    if isinstance(data, torch.Tensor): 
        data = data.detach() if detach else data
        return data.to(device, non_blocking=True)
    elif isinstance(data, (list, tuple)): 
        return type(data)(move_to_device(elem, device) for elem in data)
    elif isinstance(data, dict):
        return {key : move_to_device(value, device) for key, value in data.items()}
    else:
        return data
    
def to_python_scalar(x):
    """ 
    Extracts any one-element tensors (including those arbitrarily nested in
    dicts/lists/tuples) as Python floats and moves them to the CPU. Other
    objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        if x.numel() == 1: 
            return x.cpu().item()
        else:
            raise ValueError(
                "Valid log entries may have only 1 element but got " 
                f"{x.numel()} elements with shape {x.shape}."
            )
    elif isinstance(x, dict):
        return {k : to_python_scalar(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_python_scalar(v) for v in x)
    else:
        return x

   

