from dataclasses import is_dataclass, asdict
import importlib
import inspect
import json
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

def save_to_json(x, filepath, indent=2):
    filepath = Path(filepath)
    filepath.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(x, f, indent=indent)

def load_from_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def get_constructor_params(x):
    """ 
    `x` can be an object or class. A list of the arguments (excluding self) for
    the constructor of x's class (if an object) or of x (if a class) will be
    returned.
    """
    cls = x if isinstance(x, type) else x.__class__
    return list(inspect.signature(cls.__init__).parameters.keys())[1:] # First element is self

# ----------------------------- Pre-serialization --------------------------- #
def get_path(x):
    """ 
    x is a class (object of class 'type') or an instance of a class.
    """
    cls = x if isinstance(x, type) else x.__class__
    return cls.__module__ + '.' + cls.__qualname__

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

# --------------------------- Post-de-serialization ------------------------- #    
def is_tagged_dict(x, kind):
    """ 
    Check that x is a tagged dict of the right kind ('dataclass' or 'tensor').
    """
    if not isinstance(x, dict): return False
    if '__kind__' not in x: return False
    if x['__kind__'] != kind: return False
    return True

def load_class_from_path(path: str):
    module_path, cls_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

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


        

