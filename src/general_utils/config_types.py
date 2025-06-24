from dataclasses import asdict, dataclass, replace
from typing import Generic, TypeVar, Any, List, Optional
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
    args_cfg : Optional[ArgsConfig]
    call_locked: bool = False
    retrieve_locked: bool = False
    warn_if_locked: bool = True
    raise_exception_if_locked: bool = False

    @classmethod
    def from_callable(
        cls, 
        callable_: Any, 
        args_cfg: Optional[ArgsConfig], 
        kind: str,
        *,
        call_locked: bool = False, 
        retrieve_locked: bool = False,
        warn_if_locked: bool =True, 
        raise_exception_if_locked: bool =False
    ):
        """ 
        Central method allowing easy instantiation of a CallableConfig object 
        by passing in a reference to callable (function or class) and an
        ArgsConfig dataclass containing args for the callable.
        """
        return cls(
            path=cls._get_path(callable_, kind),
            args_cfg=args_cfg,
            call_locked=call_locked,
            retrieve_locked=retrieve_locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )
    
    @staticmethod
    def _get_path(callable_, kind):
        if kind == 'class':
            return get_cls_path(callable_)
        elif kind == 'function':
            return get_fn_path(callable_)
        else:
            raise ValueError(
                f"Unrecognized value {kind} for kind. Must be 'class' or 'function'"
            )
        
    def call(self, **kwargs) -> T:
        """ 
        Call callable using previously supplied args.
        """
        if not isinstance(self.args_cfg, ArgsConfig):
            if self.args_cfg is None:
                raise RuntimeError(
                    "The `call' method was called on this object, but args_cfg "
                    "passed in as None."
                )
            else:
                raise TypeError(
                    "Expected an ArgsConfig dataclass for self.args_cfg but " 
                    f"got type {type(self.args_cfg)}."
                )
        
        if self.call_locked:
            self._output_if_locked('call')
            return self
        
        callable_ = self.get_callable()
        return callable_(**asdict(self.args_cfg), **kwargs)
    
    def manually_call(self, **kwargs):
        self.call_locked = False
        return self.call(**kwargs)
    
    def get_callable(self):
        """ 
        Convenience method to retrieve reference to class X from the stored 
        class path.
        """
        if self.retrieve_locked:
            self._output_if_locked('retrieve')
            return self
        return load_from_path(self.path)
    
    def manually_get_callable(self):
        self.retrieve_locked = False
        return self.get_callable()
    
    def _output_if_locked(self, lock_kind):
        if lock_kind == 'call':
            message = (
                "The `call` method was called on this object with self.path=" 
                f"{self.path}, but self.call_locked=True."
            )
        elif lock_kind == 'retrieve':
            message = (
                "The `get_callable` method was called on this object with " 
                f"self.path={self.path}, but self.retrieve_locked=True."
            )
        else:
            raise ValueError(
                f"Unrecognized value for lock_kind: {lock_kind}. Must be "
                "'call' or 'retrieve'."
            )

        if self.warn_if_locked and not self.raise_exception_if_locked:
            warnings.warn(message)
        elif self.raise_exception_if_locked:
            raise RuntimeError(message)





# @dataclass
# class ClassConfig(CallableConfig):
    
#     @classmethod
#     def from_class(
#         cls, 
#         cls_obj: type, 
#         args_cfg: ArgsConfig, 
#         *,
#         locked: bool = False, 
#         warn_if_locked: bool =True, 
#         raise_exception_if_locked: bool =False
#     ):
#         """ 
#         Alias cls._from_callable.
#         """
#         return cls._from_callable(
#             cls, 
#             callable_=cls_obj,
#             args_cfg=args_cfg,
#             locked=locked,
#             warn_if_locked=warn_if_locked,
#             raise_exception_if_locked=raise_exception_if_locked
#         )
    
#     @staticmethod
#     def _get_path(cls_obj):
#         return get_cls_path(cls_obj)
    
#     def instantiate(self, **kwargs):
#         """ 
#         Alias self._call. Can set self.locked=True to prevent automatic
#         instantiation in a recursive walk (requires deferred manual
#         instantiation, see `manually_instantiate`).
#         """
#         if not self.locked:
#             return self._call(**kwargs)
#         else:
#             return self
    
#     def manually_instantiate(self, **kwargs):
#         self.lock = False
#         return self.instantiate(**kwargs)


# @dataclass
# class FunctionConfig(CallableConfig):

#     @classmethod
#     def from_fn(
#         cls, 
#         fn,
#         args_cfg: ArgsConfig, 
#         *,
#         locked: bool = False, 
#         warn_if_locked: bool =True, 
#         raise_exception_if_locked: bool =False
#     ):
#         return cls._from_callable(
#             callable_=fn,
#             args_cfg=args_cfg,
#             locked=locked,
#             warn_if_locked=warn_if_locked,
#             raise_exception_if_locked=raise_exception_if_locked
#         )

#     @staticmethod
#     def get_path(x):
#         return get_fn_path(x)
    
#     def
    
#     def execute(self, **kwargs):
#         return self._call(**kwargs)
    
#     def get_fn(self):
#         return self._get_callable()


@dataclass
class TensorArgsConfig(ArgsConfig):
    """ 
    Helper class meant to be used solely with the from_tensor convenience 
    method of the CallableConfig class.
    """
    data: List
    dtype: str
    requires_grad: bool


@dataclass
class TensorConfig(CallableConfig):

    @classmethod
    def from_tensor(
        cls, 
        t,
        locked=False,
        warn_if_locked=True,
        raise_exception_if_locked=False
        ):
        """ 
        Convenience method to instantiate a CallableConfig object from a tensor.
        """
        return cls.from_callable(
            callable_=torch.tensor,
            args_cfg=TensorArgsConfig(
                data=t.tolist(),
                dtype=str(t.dtype),
                requires_grad=t.requires_grad
            ),
            kind='function',
            call_locked=locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )
    
    def to_tensor(self):
        # Key 'dtype' points to a string, must convert to torch.dtype.
        if 'dtype' not in asdict(self.args_cfg):
            raise KeyError(
                "TensorConfig object must contain a dtype attribute for " 
                "accurate reconstruction."
            )
        try:
            dtype = getattr(torch, self.args_cfg.dtype.rsplit('.', 1)[-1])
            if not isinstance(dtype, torch.dtype):
                raise TypeError
        except (AttributeError, TypeError):
            raise ValueError(
                f"self.args_cfg.dtype = {self.args_cfg.dtype} is invalid. Must "
                "be string equivalent of valid torch.dtype, e.g. 'torch.float32."
            )
        
        updated_args_cfg = replace(self.args_cfg, dtype=dtype)

        return self.from_callable(
            callable_=torch.tensor,
            args_cfg=updated_args_cfg,
            call_locked=self.call_locked,
            retrieve_locked=self.retrieve_locked,
            warn_if_locked=self.warn_if_locked,
            raise_exception_if_locked=self.raise_exception_if_locked
        ).call()
    
    
    

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