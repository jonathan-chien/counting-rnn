import torch

from core_utils import tensor as tensor_utils

  
def validate_pmf(pmf, support_size, rtol=1e-5, atol=1e-8):
    """
    Check for valid PMF.
    
    Parameters
    ----------
    pmf (1d tensor) : Vector of probability masses.
    support_size (int or None) : Number of items in support (length of 
        `pmf`). Can be passed in as None, in which chase checking for the 
        right length of `pmf` will be skipped.
    """
    tensor_utils.validate_tensor(pmf, 1)
    if support_size is not None and len(pmf) != support_size:
        raise ValueError(
            f"pmf must be a tensor of length {support_size} but got a tensor "
            f"of length {len(pmf)}."
        )
    if not torch.isclose(torch.sum(pmf), torch.tensor([1.0]), rtol=rtol, atol=atol):
        raise ValueError(
            f"Elements of `pmf` must sum to 1, but summed to {torch.sum(pmf)}."
        )
    if not all((pmf >= 0) & (pmf <= 1)):
        raise ValueError("No value of `pmf` can be <=0 or >=1.")
        
def uniform_pmf(n, dtype=torch.float32):
    """
    Returns uniform distribution over n elements.
    """
    return torch.tile(torch.tensor([1/n], dtype=dtype), (n,))
        
def get_lexicographic_ordering(num_vars, encoding):
    """ 
    Returns
    -------
    truth_table (2d tensor): 2^(num_vars) x num_vars tensor consisting of all 
        strings in {1,-1}^(num_vars) in lexicographic order (or equivalently 
        all strings in {0,1}^num_vars.
    """
    tensor_utils.validate_tensor(encoding, 1)
    if len(encoding) != 2:
        raise RuntimeError(
            f"Expected `encoding` to have length 2 but got length {len(encoding)}."
        )

    truth_table = torch.full((2**num_vars, num_vars), torch.nan)
    for i_var in range(num_vars):
        truth_table[:, i_var] = torch.tile(
            torch.repeat_interleave(
                encoding, 
                2**(num_vars - (i_var+1))
            ), 
            (2**i_var,)
        )

    return truth_table

def convert_encoding_01_to_11(tensor):
    """Convert {0, 1} encoding to {1, -1} encoding."""
    return  1 - 2 * tensor

def convert_encoding_11_to_01(tensor):
    """Convert {1, -1} encoding to {0, 1} encoding."""
    return (tensor / -2 + 0.5).to(torch.int16)

def convert_encoding_general(tensor, new_encoding):
    """
    Convert between any two binary encodings.

    Parameters
    ----------
    tensor (3d tensor) : Binary torch tensor with values in the set {a, b}, 
        where a < b.
    new_encoding (sequence) : Two element sequence, e.g. [c, d]. All `a`
        in `tensor` are mapped to min(c, d), and all `b` in `tensor` are mapped
        to max(c, d).

    Returns
    -------
    new_tensor (3d tensor) : Binary torch tensor with same shape as 
        `tensor` but with values reassigned as described above.
    """
    if not len(new_encoding) == 2: 
        raise RuntimeError("`new_encoding` should be a sequence of length 2.")
    
    a = torch.min(tensor)
    b = torch.max(tensor)
    new_tensor = torch.full(tensor.shape, torch.nan)
    new_tensor[tensor == a] = torch.min(new_encoding)
    new_tensor[tensor == b] = torch.max(new_encoding)

    return new_tensor




