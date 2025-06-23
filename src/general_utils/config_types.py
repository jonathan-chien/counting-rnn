from dataclasses import asdict, dataclass, is_dataclass, fields
from typing import Generic, TypeVar, Any, List
import warnings
warnings.simplefilter("always")

import torch

from . serialization import get_cls_path, get_fn_path, load_from_path


T = TypeVar('T')


@dataclass
class ArgsConfig:
    """Type I"""
    pass


@dataclass 
class ContainerConfig:
    """Type II"""
    pass


@dataclass
class CallableConfig(Generic[T]):
    """ 
    Utility dataclass for storing a configuration of an object of an arbitrary
    class X within a dataclass config hierarchy. The class path for X is stored
    as a string and constructor args are stored in a dataclass; thus, this
    dataclass can easily be converted to a tagged dict and serialized with
    json, and the object of class X can be instantiated following
    de-serialization. 

    Type III
    """
    path : str
    args_cfg : ArgsConfig 
    kind: str
    locked: bool = False
    warn_if_locked: bool = True
    raise_exception_if_locked: bool = False

    @classmethod
    def from_callable(
        cls, 
        callable_: Any, 
        args_cfg: ArgsConfig, 
        kind: str,
        *,
        locked: bool =False, 
        warn_if_locked: bool =True, 
        raise_exception_if_locked: bool =False
    ):
        """ 
        Central method allowing easy instantiation of a FactoryConfig object by
        passing in a reference to callable (function or class) and an
        ArgsConfig dataclass containing args for the callable.
        """
        return cls(
            path=cls.get_path(callable_, kind),
            args_cfg=args_cfg,
            kind=kind, # Register kind
            locked=locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )
    
    @staticmethod
    def get_path(callable_, kind):
        if kind == 'class':
            return get_cls_path(callable_)
        if kind == 'function': 
            return get_fn_path(callable_)
        else:
            raise ValueError(
                f"Unrecognized value {kind} for `kind`. Must be 'class' or 'function'."
            )
        
    def call(self, **kwargs) -> T:
        """ 
        Call callable using previously supplied args.
        """
        if self.locked:
            message = (
                "The `call` method was called on this object with self.path=" 
                f"{self.path}, but self.locked=True."
            )
            if self.warn_if_locked and not self.raise_exception_if_locked:
                warnings.warn(message)
            elif self.raise_exception_if_locked:
                raise RuntimeError(message)
            return self # y = x.call() results in y == x
         
        if not isinstance(self.args_cfg, ArgsConfig):
            raise TypeError(
                "Expected an ArgsConfig dataclass for self.args_cfg but got " 
                f"type {type(self.args_cfg)}."
            )
        
        callable_ = self.get_callable()
        return callable_(**asdict(self.args_cfg), **kwargs)
    
    def get_callable(self):
        """ 
        Convenience method to retrieve reference to class X from the stored 
        class path.
        """
        return load_from_path(self.path)
    
   
    

# @dataclass
# class FunctionConfig:
#     fn_path: str
#     args_cfg: ArgsConfig
#     locked: bool = False
#     warn_if_locked: bool = True
#     raise_exception_if_locked = False

#     @classmethod
#     def from_fn(
#         cls, 
#         fn, 
#         args_cfg, 
#         locked=False, 
#         warn_if_locked=True, 
#         raise_exception_if_locked=False
#     ):
#         return cls(
#             fn_path=get_fn_path(fn),
#             args_cfg=args_cfg,
#             locked=locked,
#             warn_if_locked=warn_if_locked,
#             raise_exception_if_locked=raise_exception_if_locked
#         )
    
#     def execute(self, **kwargs):
#         """ 
#         """
#         if self.locked:
#             if self.warn_if_locked and not self.raise_exception_if_locked:
#                 warnings.warn(
#                     "The instantiate method was called on this object, but self.locked=True."
#                 )
#             elif self.raise_exception_if_locked:
#                 raise RuntimeError(
#                     "The instantiate method was called on this object, but self.locked=True."
#                 )
#             return self # y = x.instantiate() results in y == x
         
#         if not isinstance(self.args_cfg, ArgsConfig):
#             raise TypeError(
#                 "Expected an ArgsConfig dataclass for self.args_cfg but got " 
#                 f"type {type(self.args_cfg)}."
#             )
        
#         fn = self.get_fn()
#         return fn(**asdict(self.args_cfg), **kwargs)
    
#     def get_fn(self):
#         """ 
#         """
#         return load_from_path(self.fn_path)
    

# @dataclass
# class TensorArgs(ArgsConfig):
#     """ 
#     """
#     data: List
#     dtype: str
#     requires_grad: bool


# @dataclass
# class TensorConfig(FunctionConfig):
#     """ 
#     """
#     @classmethod
#     def from_tensor(
#         cls, 
#         t,
#         locked=False,
#         warn_if_locked=True,
#         raise_exception_if_locked=False
#         ):
#         return cls.from_fn(
#             fn=torch.tensor,
#             args_cfg=ArgsConfig(
#                 data=t.tolist(),
#                 dtype=str(t.dtype),
#                 requires_grad=t.requires_grad
#             )
#         )