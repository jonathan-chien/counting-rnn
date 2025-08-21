from dataclasses import fields, is_dataclass, replace
import importlib
import inspect
import types
from typing import Union, get_args, get_origin

import torch

from . import fileio as fileio_utils
from . import recursion as recursion_utils
from . import validation as validation_utils


# ------------------------------- General ------------------------------------ #
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


# --------------------------- Recursion branches ----------------------------- #
# These are used for both serialization and de-serialization.

# --------------------------------------------------
def handle_dict_with_transform_to_dataclass(d, recurse):
    """ 
    This function breaks the pattern of separating branch conditions from
    transformations, as a transformation is applied here after branching.
    However, since we wish both to descend recursively into tagged dicts and to
    transform them into dataclasses, it seems there is no way to keep these
    separate without compromising the integrity and safety of the `recursive`
    function itself.
    """
    d = {k : recurse(v) for k, v in d.items()}
    x = tagged_dict_to_dataclass_instance(d)
    # x = serialization_utils.tagged_dict_to_tensor(x)
    # x = serialization_utils.tagged_dict_to_function(x)
    return x

dict_branch_with_transform_to_dataclass = (validation_utils.is_dict, handle_dict_with_transform_to_dataclass)

# --------------------------------------------------
def handle_dataclass_with_transform_to_dict(d, recurse):
    """ 
    This function breaks the pattern of separating branch conditions from leaf
    transformations, as a transformation is applied here after branching.
    See documentation for handle_dict_with_transform, as this siutation is the 
    dual of the one addressed there.

    NB: Attempting to reconstruct the dataclass from the result of the dict
    comprehension and call a utility from io_utils to convert the dataclass to
    a tagged dict will likely result in failed validation for the dataclass
    upon attempted reconstruction, as some of its fields may have already been
    modified at a deeper recursion level. Instead, this is done manually here.
    """
    # dataclass = type(d)(**{f.name : recurse(getattr(d, f.name)) for f in fields(d)})
    # return io_utils.dataclass_instance_to_tagged_dict(dataclass)
    dataclass_as_dict = {f.name: recurse(getattr(d, f.name)) for f in fields(d)}
    return {
        **dataclass_as_dict,
        '__path__' : get_cls_path(d),
        '__kind__' : 'dataclass'
    }

dataclass_branch_with_transform_to_dict = (is_dataclass, handle_dataclass_with_transform_to_dict)

# --------------------------------------------------
# def is_dtype_annotation(annotation):
#     # Return true if annotation is just torch.dtype.
#     if annotation is torch.dtype:
#         return True
    
#     # Else, return true if is Union and any element is torch.dtype.
#     origin = get_origin(annotation)
#     if origin is Union: # Optional[x] ~ Union[x, None]
#         return any(elem is torch.dtype for elem in get_args(annotation))
    
#     return False

# def get_torch_dtype_from_str(s: str):
#     if not isinstance(s, str):
#         return s
    
#     dtype_name = s.split('.')[-1]
#     try:
#         dtype = getattr(torch, dtype_name)
#     except AttributeError:
#         raise ValueError(f"Invalid torch.dtype string: {s}.")

#     if isinstance(dtype, torch.dtype):
#         return dtype
#     else:
#         raise ValueError(
#             f"torch.dtype string {s} could not be resolved to a valid torch.dtype."
#         )
    
def handle_dataclass_with_factory_config(d, recurse):
    # Global import on 06/21/25 causes circular import error.
    from .config import FactoryConfig

    # Descend recursively into dataclass first.
    d_transformed = recursion_utils.handle_dataclass(d, recurse)

    # # Convert dtype strings to torch.dtypes.
    # updates = {}
    # for f in fields(d_transformed):
    #     value = getattr(d_transformed, f.name)
    #     if is_dtype_annotation(f.type) and isinstance(value, str):
    #         updates[f.name] = get_torch_dtype_from_str(value)
    # if updates:
    #     d_transformed = replace(d_transformed, **updates)

    # Better to check this here, so that this condition is always checked for
    # any dataclass. Otherwise, if separate branch conditionals are used in the
    # recursive function, and dataclass_branch comes first, the conditional
    # check loop will short circuit, and deeper items may not be reached.
    if isinstance(d_transformed, FactoryConfig):
        return d_transformed.recover()
    else:
        return d_transformed

dataclass_branch_with_factory_config = (is_dataclass, handle_dataclass_with_factory_config)


# ------------------------------- Serialization ------------------------------ #
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
            dataclass_branch_with_transform_to_dict
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
            dict_branch_with_transform_to_dataclass,
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
            dataclass_branch_with_factory_config
        ),
        leaf_fns=(
            lambda x: x,
        )
    )