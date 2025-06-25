from dataclasses import asdict, dataclass, replace
from typing import Generic, TypeVar, Any, Dict, List, Literal, Optional
import warnings
warnings.simplefilter("always")

import torch

from . serialization import get_cls_path, get_fn_path, load_from_path


# ------------------------------ Config types ------------------------------- #
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
class FactoryConfig(Generic[T]):
    """
    Since TensorConfig subclasses CallableConfig, all instances of the former
    are technically instances of the latter. However, this may lead to
    confusion if, e.g., a function like is_callable_config is called. This
    should return True for TensorConfig objects, but this may not be the most
    intuitive interpretation of that name. For improved readability,
    CallableConfig and thus all of its child classes subclass the FactoryConfig
    base class, which can be reference in isinstance checks. These collectively
    constitute the Type III dataclasses.
    """
    pass

@dataclass
class CallableConfig(FactoryConfig[T]):
    """ 
    Utility dataclass for storing 
    """
    path : str
    args_cfg : Optional[ArgsConfig]
    kind: Literal['class', 'function']
    recovery_mode: Literal['call', 'get_callable']
    locked: bool = False
    warn_if_locked: bool = True
    raise_exception_if_locked: bool = False

    @classmethod
    def from_callable(
        cls, 
        callable_: Any, 
        args_cfg: Optional[ArgsConfig], 
        kind: str,
        recovery_mode: str,
        *,
        locked: bool = False, 
        warn_if_locked: bool =True, 
        raise_exception_if_locked: bool =False
    ):
        """ 
        """
        return cls(
            path=cls._get_path(callable_, kind),
            args_cfg=args_cfg,
            kind=kind,
            recovery_mode=recovery_mode,
            locked=locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )
    
    @staticmethod
    def _get_path(callable_, kind):
        """ 
        """
        if kind == 'class':
            return get_cls_path(callable_)
        elif kind == 'function':
            return get_fn_path(callable_)
        else:
            raise ValueError(
                f"Unrecognized value {kind} for kind. Must be 'class' or 'function'"
            )
        
    def _call(self, **kwargs) -> T:
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
        
        callable_ = self._get_callable()
        return callable_(**asdict(self.args_cfg), **kwargs)
    
    # def manually_call(self, **kwargs):
    #     self.call_locked = False
    #     return self.call(**kwargs)
    
    def _get_callable(self):
        """ 
        Convenience method to retrieve reference to class X from the stored 
        class path.
        """
        return load_from_path(self.path)
    
    # def manually_get_callable(self):
    #     self.retrieve_locked = False
    #     return self.get_callable()
    
    def _output_if_locked(self):
        """ 
        """
        message = (
            "The `recover` method was called on this object with self.path=" 
            f"{self.path}, but self.locked=True."
        )
        if self.warn_if_locked and not self.raise_exception_if_locked:
            warnings.warn(message)
        elif self.raise_exception_if_locked:
            raise RuntimeError(message)

    def recover(self, **kwargs):
        """ 
        """
        if self.locked:
            self._output_if_locked()
            return self
        elif self.recovery_mode == 'call':
            print(self.path) # Development/debugging
            return self._call(**kwargs)
        elif self.recovery_mode == 'get_callable':
            if len(kwargs) != 0:
                raise RuntimeError(
                    "It seems kwargs were passed to the `recover` method. Since " 
                    "self.recovery_mode='get_callable', `recover` is a wrapper " 
                    "to `_get_callable`, which takes no arguments."
                )
            print(self.path) # Development/debugging
            return self._get_callable()
        else:
            raise RuntimeError(
                "Unexpected condition reached during retrieval process, " 
                "liked due to unrecognized value for self.recovery_mode: "
                f"'{self.recovery_mode}'. Must be 'call' or 'get_callable'."
            )
        
    def manually_recover(self, **kwargs):
        """ 
        """
        self.locked = False
        return self.recover(**kwargs)



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
            recovery_mode='call',
            locked=locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )
    
    def recover(self):
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
        temporary_tensor_config = type(self).from_callable(
            callable_=torch.tensor,
            args_cfg=updated_args_cfg,
            kind=self.kind,
            recovery_mode=self.recovery_mode,
            locked=self.locked,
            warn_if_locked=self.warn_if_locked,
            raise_exception_if_locked=self.raise_exception_if_locked
        )
        return super(TensorConfig, temporary_tensor_config).recover()
    
    
    

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

# ---------------------------- Reproducibility ------------------------------ #
@dataclass
class SeedConfig(ArgsConfig):
    torch_seed: int
    cuda_seed: int


@dataclass
class TorchDeterminismConfig(ArgsConfig):
    use_deterministic_algos: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True


@dataclass
class ReproducibilityConfig(ContainerConfig):
    entropy: int
    seed_cfg_dict: Dict[str, SeedConfig]
    torch_determinisim_cfg_dict : Dict[str, TorchDeterminismConfig]