from dataclasses import dataclass, is_dataclass, asdict
import importlib
import inspect
import types
from typing import Any

import torch

from . import recursion as r_utils
from . import fileio as io_utils


def get_constructor_params(x):
    """ 
    `x` can be an object or class. A list of the arguments (excluding self) for
    the constructor of x's class (if an object) or of x (if a class) will be
    returned.
    """
    cls = x if isinstance(x, type) else x.__class__
    return list(inspect.signature(cls.__init__).parameters.keys())[1:] # First element is self

def serialize_and_deserialize(cfg, filepath):
    """ 
    """
    # Convert for serialization. TODO: implement recursive check for dicts, return Boolean, could allow file deletion if dicts don't match
    serializable_cfg_dict = r_utils.recursive(
        cfg,
        branch_conditionals=(
            r_utils.dict_branch, 
            r_utils.tuple_branch, 
            r_utils.list_branch, 
            r_utils.dataclass_branch_with_transform
        ),
        leaf_fns=(
            tensor_to_tagged_dict,
            function_to_tagged_dict
        )
    )

    # Serialize/save.
    io_utils.save_to_json(serializable_cfg_dict, filepath, indent=2)

    # Deserialize and reconstruct.
    deserialized_cfg_dict = io_utils.load_from_json(filepath)
    reconstructed_cfg = r_utils.recursive(
        deserialized_cfg_dict,
        branch_conditionals=(
            r_utils.dict_branch_with_transform,
            r_utils.tuple_branch, 
            r_utils.list_branch, 
        ),
        leaf_fns=(
            lambda x: x,
        )
    )

    return reconstructed_cfg

# ----------------------------- Pre-serialization --------------------------- #
def get_cls_path(x):
    """ 
    x is a class (object of class 'type') or an instance of a class.
    """
    cls = x if isinstance(x, type) else x.__class__
    return get_import_path(cls)

def get_fn_path(x):
    """ 
    """
    if not isinstance(x, types.FunctionType):
        raise TypeError(f"Expected a function, but got type {type(x)}.")
    return get_import_path(x)

def get_import_path(x):
    """ 
    """
    return x.__module__ + '.' + x.__qualname__

def dataclass_instance_to_tagged_dict(x):
    """ 
    Utility for converting dataclass instances to a tagged dict.
    """
    if is_dataclass(x):
        d = asdict(x)
        d['__path__'] = get_cls_path(x)
        d['__kind__'] = 'dataclass'
        return d
    else:
        return x

def tensor_to_tagged_dict(x): 
    """ 
    Converts input to a tagged dict if a tensor, otherwise returns input 
    unchanged (this is necessary for this function to be used as a leaf_fn
    with the recursive function from the recursion module).
    """
    if isinstance(x, torch.Tensor):
        return {
            '__path__' : 'torch.Tensor',
            '__kind__' : 'tensor',
            'data' : x.tolist(),
            'dtype' : str(x.dtype),
            'requires_grad' : x.requires_grad
        }
    else:
        return x
    
def function_to_tagged_dict(f):
    if isinstance(f, types.FunctionType):
        return {
            '__path__' : get_fn_path(f),
            '__kind__' : 'function',
        }
    else:
        return f


# --------------------------- Post-de-serialization ------------------------- #    
def is_tagged_dict(x, kind):
    """ 
    Check that x is a tagged dict of the right kind ('dataclass' or 'tensor').
    """
    if not isinstance(x, dict): return False
    if '__kind__' not in x: return False
    if x['__kind__'] != kind: return False
    return True

def load_from_path(path: str):
    """ 
    """
    module_path, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

def tagged_dict_to_dataclass_instance(x):
    """ 
    """
    if is_tagged_dict(x, 'dataclass'):
        cls = load_from_path(x['__path__'])
        param_names = get_constructor_params(cls)
        args = {key: val for key, val in x.items() if key in param_names}
        return cls(**args)
    else:
        return x
    
def tagged_dict_to_tensor(x):
    """ 
    """
    if is_tagged_dict(x, 'tensor'):
        # Filter out tags.
        args = {
            key: val for key, val in x.items() 
            if key not in ('__path__', '__kind__')
        }

        # Key 'dtype' points to a string, must convert to torch.dtype.
        if 'dtype' not in args:
            raise KeyError(
                "Tagged tensor dict must contain a dtype key for accurate reconstruction."
            )
        try:
            dtype = getattr(torch, args['dtype'].rsplit('.', 1)[-1])
            if not isinstance(dtype, torch.dtype):
                raise TypeError
            args['dtype'] = dtype
        except (AttributeError, TypeError):
            raise ValueError(
                f"args['dtype'] = {args['dtype']} yielded invalid dtype string "
                f"{dtype}, must be string equivalent of valid torch.dtype, "
                "e.g. 'torch.float32."
            )
        
        return torch.tensor(**args)
    else:
        return x
    
def tagged_dict_to_function(x):
    """ 
    """
    if is_tagged_dict(x, 'function'):
        return load_from_path(x['__path__'])
    else:
        return x
    
def recursive_instantiation(x):
    """ 
    Utility to recursively walk through nested object and replace any 
    FactoryConfig objects with the instantiated object of the associated class.
    """
    return r_utils.recursive(
        x,
        branch_conditionals=(
            r_utils.dict_branch,
            r_utils.list_branch,
            r_utils.tuple_branch,
            r_utils.dataclass_branch_with_instantiation
        ),
        leaf_fns=(
            lambda x: x,
        )
    )