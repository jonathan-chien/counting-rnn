from dataclasses import is_dataclass, fields
from . import serialization as ser_utils


def recursive(x, branch_conditionals, leaf_fns, depth=None, max_depth=100):
    """ 
    """
    # Custom recursion limiter.
    depth = 0 if depth is None else depth + 1
    if depth == max_depth:
        raise RecursionError(
            f"User-specified max recursion depth of {max_depth} exceeded."
        )

    # Recursive descent. Transformations in consequent possible here as well.
    for antecedent, consequent in branch_conditionals:
        if antecedent(x):
            return consequent(
                x, lambda x: recursive(x, branch_conditionals, leaf_fns, depth)
            )

    # Apply transformations to leaf cases.
    for fn in leaf_fns: 
        x = fn(x)

    return x

def is_dict(x):
    return isinstance(x, dict)

def handle_dict(d, recurse):
    return {k : recurse(v) for k, v in d.items()}

dict_branch = (is_dict, handle_dict)

def handle_dict_with_transform(d, recurse):
    """ 
    This function breaks the pattern of separating branch conditions from
    transformations, as a transformation is applied here after branching.
    However, since we wish both to descend recursively into tagged dicts and to
    transform them into dataclasses, it seems there is no way to keep these
    separate without compromising the integrity and safety of the `recursive`
    function itself. This solution confines the breaking of the separation of
    branches and leaves to this function, while preserving the `recursive` 
    function's generality.
    """
    d = {k : recurse(v) for k, v in d.items()}
    x = ser_utils.tagged_dict_to_dataclass_instance(d)
    x = ser_utils.tagged_dict_to_tensor(x)
    x = ser_utils.tagged_dict_to_function(x)
    return x

dict_branch_with_transform = (is_dict, handle_dict_with_transform)

def is_list(x):
    return isinstance(x, list)

def handle_list(l, recurse):
    return [recurse(v) for v in l]

list_branch = (is_list, handle_list)

def is_tuple(x):
    return isinstance(x, tuple)

def handle_tuple(t, recurse):
    return tuple(recurse(v) for v in t)

tuple_branch = (is_tuple, handle_tuple)

def handle_dataclass(d, recurse):
    """ 
    """
    return type(d)(**{f.name : recurse(getattr(d, f.name)) for f in fields(d)})

dataclass_branch = (is_dataclass, handle_dataclass)

def handle_dataclass_with_transform(d, recurse):
    """ 
    This function breaks the rule of separating branch conditions from leaf
    transformations, as a transformation is applied here after branching.
    See documentation for handle_dict_with_transform, as this problem is the 
    dual of the one addressed there.

    NB: Attempting to reconstruct the dataclass from the result of the dict
    comprehension and call a utility from io_utils to convert the dataclass to
    a tagged dict will likely result in failed validation for the dataclass
    upon attempted reconstruction, as some of its fields may have already been
    modified at a deeper recursion level. Instead, this is done manually here.
    """
    # dataclass = type(d)(**{f.name : recurse(getattr(d, f.name)) for f in fields(d)})
    # return io_utils.dataclass_instance_to_tagged_dict(dataclass)
    dataclass_as_dict = {f.name : recurse(getattr(d, f.name)) for f in fields(d)}
    return {
        **dataclass_as_dict,
        '__path__' : ser_utils.get_cls_path(d),
        '__kind__' : 'dataclass'
    }

dataclass_branch_with_transform = (is_dataclass, handle_dataclass_with_transform)

def handle_dataclass_with_call(d, recurse):
    """ 
    """
    # Global import on 06/21/25 causes circular import error.
    from .config_types import CallableConfig

    # Descend recursively into dataclass first.
    dataclass = type(d)(**{f.name : recurse(getattr(d, f.name)) for f in fields(d)})

    if isinstance(dataclass, CallableConfig):
        return dataclass.call()
    else:
        return dataclass

dataclass_branch_with_instantiation = (is_dataclass, handle_dataclass_with_call)

