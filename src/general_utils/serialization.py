from dataclasses import dataclass, is_dataclass, asdict
import importlib
import inspect
from typing import Any

import torch


@dataclass
class ClassConfig:
    """ 
    Utility dataclass for storing a configuration of an object of an arbitrary
    class X within a dataclass config hierarchy. The class path for X is stored
    as a string and constructor args are stored in a dataclass; thus, this
    dataclass can easily be converted to a tagged dict and serialized with
    json, and the object of class X can be instantiated following
    de-serialization. 
    """
    cls_path : str
    constructor_args : Any # Should be a dataclass

    @classmethod
    def from_class(cls, cls_obj, constructor_args):
        """ 
        Convenience method allowing easy instantiation of a ClassConfig object 
        by passing in a reference to class X and a dataclass containing 
        constructor args for X.
        """
        return cls(
            cls_path=get_path(cls_obj),
            constructor_args=constructor_args
        )

    def instantiate(self):
        """ 
        Instantiate object of class X using previously supplied constructor args.
        """
        if not is_dataclass(self.constructor_args):
            raise TypeError(
                "Expected a dataclass for self.constructor_args but got type " 
                f"{type(self.constructor_args)}."
            )
        cls = self.get_class()
        return cls(**asdict(self.constructor_args))
    
    def get_class(self):
        """ 
        Convenience method to retrieve reference to class X from the stored 
        class path.
        """
        return load_class_from_path(self.cls_path)
    

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