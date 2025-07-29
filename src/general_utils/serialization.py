from dataclasses import fields, is_dataclass
import importlib
import inspect
import types

import torch

from . import recursion as recursion_utils
from . import fileio as fileio_utils


def get_constructor_params(x):
    """ 
    `x` can be an object or class. A list of the arguments (excluding self) for
    the constructor of x's class (if an object) or of x (if a class) will be
    returned.
    """
    cls = x if isinstance(x, type) else x.__class__
    return list(inspect.signature(cls.__init__).parameters.keys())[1:] # First element is self

def shallow_asdict(d): 
    """ 
    """
    if not is_dataclass(d):
        raise TypeError("`shallow_asdict` should be called on dataclass instances.")
    return {f.name : getattr(d, f.name) for f in fields(d)}


# ------------------------------ Serialization ------------------------------ #
def get_cls_path(x):
    """ 
    x is a class (object of class 'type') or an instance of a class.
    """
    cls = x if isinstance(x, type) else x.__class__
    return get_import_path(cls)

def get_fn_path(x):
    """ 
    """
    if not isinstance(x, (types.FunctionType, types.BuiltinFunctionType)):
        raise TypeError(f"Expected a function, but got type {type(x)}.")
    return get_import_path(x)

def get_import_path(x):
    """ 
    """
    if x is torch.tensor: # Special case, else would get 'torch._VariableFunctionsClass.tensor'.
        return 'torch.tensor'
    return x.__module__ + '.' + x.__qualname__

def dataclass_instance_to_tagged_dict(x):
    """ 
    Utility for converting dataclass instances to a tagged dict.
    """
    if is_dataclass(x):
        d = shallow_asdict(x)
        d['__path__'] = get_cls_path(x)
        d['__kind__'] = 'dataclass'
        return d
    else:
        return x
    
def recursive_dataclass_to_tagged_dict(x):
    """ 
    Takes nested structure dataclasses and converts all dataclasses to tagged
    dicts.
    """
    # Convert for serialization. TODO: implement recursive check for dicts, return Boolean, could allow file deletion if dicts don't match
    return recursion_utils.recursive(
        x,
        branch_conditionals=(
            recursion_utils.tuple_branch, 
            recursion_utils.list_branch, 
            recursion_utils.dataframe_branch,
            recursion_utils.dict_branch, 
            recursion_utils.dataclass_branch_with_transform_to_dict
        ),
        leaf_fns=(
            lambda a: a,
        )
    )
    
def serialize(cfg, filepath):
    """ 
    """
    # # Convert for serialization. TODO: implement recursive check for dicts, return Boolean, could allow file deletion if dicts don't match
    # serializable_cfg = recursion_utils.recursive(
    #     cfg,
    #     branch_conditionals=(
    #         recursion_utils.dict_branch, 
    #         recursion_utils.tuple_branch, 
    #         recursion_utils.list_branch, 
    #         recursion_utils.dataclass_branch_with_transform_to_dict
    #     ),
    #     leaf_fns=(
    #         lambda x: x,
    #     )
    # )
    # TODO: Could add call to tensor_utils.recursive_tensor_to_tensor_config here and just write tensors in configure_* scripts.
    serializable = recursive_dataclass_to_tagged_dict(cfg)

    # Serialize/save.
    fileio_utils.save_to_json(serializable, filepath, indent=2)
    return serializable


# ----------------------------- De-serialization ---------------------------- #    
def is_tagged_dict(x, kind):
    """ 
    Check that x is a tagged dict of the right kind ('dataclass' or 'tensor').
    """
    if not isinstance(x, dict): return False
    if '__kind__' not in x: return False
    if x['__kind__'] != kind: return False
    return True

# def load_from_path(path: str):
#     """ 
#     """
#     module_path, cls_name = path.rsplit('.', 1)
#     module = importlib.import_module(module_path)
#     return getattr(module, cls_name)

def load_from_path(path: str, module_depth: int = 1):
    """ 
    """
    path_parts = path.rsplit('.')

    # Validate.
    if module_depth >= len(path_parts):
        raise ValueError(
            "`module_depth` must be strictly less than the number of path " 
            f"parts ({len(path_parts)}), but got {module_depth}."
        )
    
    # Reconstruct the two parts: path to module and remainder of path.
    module_path = '.'.join(path_parts[:-module_depth])
    module = importlib.import_module(module_path)
    attr_path = path_parts[-module_depth:]
    
    # Traverse to target attribute.
    obj = module
    for attr in attr_path:
        obj = getattr(obj, attr)

    return obj

def tagged_dict_to_dataclass_instance(x):
    """ 
    """
    if is_tagged_dict(x, 'dataclass'):
        cls = load_from_path(x['__path__'], module_depth=1)
        param_names = get_constructor_params(cls)
        args = {key: val for key, val in x.items() if key in param_names}
        return cls(**args)
    else:
        return x
    
def recursive_tagged_dict_to_dataclass(x):
    return recursion_utils.recursive(
        x,
        branch_conditionals=(
            recursion_utils.tuple_branch, 
            recursion_utils.list_branch, 
            recursion_utils.dataframe_branch,
            recursion_utils.dict_branch_with_transform_to_dataclass,
        ),
        leaf_fns=(
            lambda x: x,
        )
    )
    
def deserialize(filepath):
    """ 
    """
    # Deserialize and reconstruct.
    serializable = fileio_utils.load_from_json(filepath)

    # return recursion_utils.recursive(
    #     serializable,
    #     branch_conditionals=(
    #         recursion_utils.dict_branch_with_transform_to_dataclass,
    #         recursion_utils.tuple_branch, 
    #         recursion_utils.list_branch, 
    #     ),
    #     leaf_fns=(
    #         lambda x: x,
    #     )
    # )
    return recursive_tagged_dict_to_dataclass(serializable)
    
def recursive_recover(x):
    """ 
    Utility to recursively walk through nested object and replace any FactoryConfig
    objects with the result of calling the `recover` method on that object.
    """
    return recursion_utils.recursive(
        x,
        branch_conditionals=(
            recursion_utils.list_branch,
            recursion_utils.tuple_branch,
            recursion_utils.dataframe_branch,
            recursion_utils.dict_branch,
            recursion_utils.dataclass_branch_with_factory_config
        ),
        leaf_fns=(
            lambda x: x,
        )
    )