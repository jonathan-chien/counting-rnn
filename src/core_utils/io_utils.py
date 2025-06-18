from dataclasses import is_dataclass, asdict
import importlib
import inspect
import os
from pathlib import Path
from typing import Callable, Union

import torch


# ---------------------------- General utilities ---------------------------- #
def make_dir(*path_parts: Union[str, Path], chdir=False):
    dir = Path(*path_parts) 
    dir.mkdir(parents=True, exist_ok=True)
    if chdir: os.chdir(dir)
    return dir

def get_path(x):
    """ 
    x is a class (object of class 'type') or an instance of a class.
    """
    cls = x if isinstance(x, type) else x.__class__
    return cls.__module__ + '.' + cls.__qualname__

def get_constructor_params(x):
    """ 
    `x` can be an object or class. A list of the arguments (excluding self) for
    the constructor of x's class (if an object) or of x (if a class) will be
    returned.
    """
    cls = x if isinstance(x, type) else x.__class__
    return list(inspect.signature(cls.__init__).parameters.keys())[1:] # First element is self

def load_class_from_path(path: str):
    module_path, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

def is_tagged_dict(x, kind):
    """ 
    Check that x is a tagged dict of the right kind ('dataclass' or 'tensor').
    """
    if not isinstance(x, dict): return False
    if '__kind__' not in x: return False
    if x['__kind__'] != kind: return False
    return True






# Serialization -----------------------
# ENCODERS = []

# def register_encoder(antecedent: Callable):
#     """ 
#     Decorator to register antecedent condition check (e.g. is_dataclass) with
#     a consequent (e.g. dataclass_to_tagged_dict).
#     """
#     def decorator(consequent: Callable):
#         ENCODERS.append((antecedent, consequent))
#         return consequent
#     return decorator

# @register_encoder(lambda x: is_dataclass(x) and not isinstance(x, type))
def dataclass_instance_to_tagged_dict(x):
    """ 
    Utility for converting dataclass instances to a tagged dict.
    """
    if is_dataclass(x):
        d = asdict(x)
        d['__path__'] = get_path(x)
        d['__kind__'] = 'dataclass'
        return d
    else:
        return x

# @register_encoder(lambda x: isinstance(x, type))
# def class_to_tagged_dict(x):
#     return {
#         '__path__' : get_path(x),
#         '__kind__' : 'class'
#     }

# @register_encoder(lambda x: isinstance(x, torch.Tensor))
def tensor_to_tagged_dict(x): 
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

# def distribution_to_tagged_dict(x):
#     pass
    

# Deserialization ------------------------
# def tagged_dict_to_instance(x):
#     """ 
#     """
#     if isinstance(x, dict) and '__path__' in x:
#         cls = load_class_from_path(x['__path__'])
#         param_names = get_constructor_params(cls)
#         args = {name: val for name, val in x.items() if name in param_names}
#         return cls(**args)
#     else:
#         return x

# def tagged_dict_to_instance(x):
#     if isinstance(x, dict) and '__path__' in x and x['__kind__'] == 'instance':
#         cls_path = x['__path__']
#         cls = load_class_from_path(cls_path)
#         args = {k : v for k, v in x.items() if k != '__path__'} # Extract args w/o mutation
#         return cls(**args)
    
# def tagged_dict_to_class(x):
#     if isinstance(x, dict) and '__path__' in x and x['__kind__'] == 'instance':
#         cls_path = x['__path__']
#         return load_class_from_path(cls_path)
#     else:
#         return x
    
def tagged_dict_to_dataclass_instance(x):
    if is_tagged_dict(x, 'dataclass'):
        cls = load_class_from_path(x['__path__'])
        param_names = get_constructor_params(cls)
        args = {key: val for key, val in x.items() if key in param_names}
        return cls(**args)
    else:
        return x
    
def tagged_dict_to_tensor(x):
    if is_tagged_dict(x, 'tensor'):
        args = {
            key: val for key, val in x.items() 
            if key not in ('__path__', '__kind__')
        }
        return torch.tensor(**args)
    else:
        return x


        

