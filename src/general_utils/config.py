from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Union
import warnings
warnings.simplefilter("always")

import torch

from . serialization import get_cls_path, get_fn_path, load_from_path, shallow_asdict


# ------------------------------ Config types ------------------------------- #
@dataclass
class ArgsConfig:
    """
    Type I
    
    Attribute names should exactly match parameter names for function or method.
    """
    pass


@dataclass 
class ContainerConfig:
    """
    Type II

    Attributes are heterogeneous, can contain other config dataclasses.
    """
    pass

@dataclass
class FactoryConfig:
    """
    Type III

    Since TensorConfig subclasses CallableConfig, all instances of the former
    are instances of the latter. However, this may lead to confusion if, e.g.,
    a function like is_callable_config is called. This should return True for
    TensorConfig objects, but this may not necessarily be intuitive, as
    CallableConfig and TensorConfig objects are often used side by side. For
    improved readability, CallableConfig and thus all of its child classes
    subclass the FactoryConfig base class, which can be referenced in
    isinstance checks. These collectively constitute the Type III dataclasses.
    """
    pass


@dataclass
class CallableConfig(FactoryConfig):
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
        elif kind in ('function', 'static_method'):
            return get_fn_path(callable_)
        else:
            raise ValueError(
                f"Unrecognized value {kind} for kind. Must be 'class' or 'function'."
            )
        
    def _call(self, **kwargs):
        """ 
        Call callable using previously supplied args.
        """
        if not isinstance(self.args_cfg, ArgsConfig):
            if self.args_cfg is None:
                raise RuntimeError(
                    "The `_call' method was called on this object, but args_cfg "
                    "passed in as None."
                )
            else:
                raise TypeError(
                    "Expected an ArgsConfig dataclass for self.args_cfg but " 
                    f"got type {type(self.args_cfg)}."
                )
        
        callable_ = self._get_callable()
        return callable_(**{**shallow_asdict(self.args_cfg), **kwargs})
    
    def _get_callable(self):
        """ 
        Retrieve reference to callable from the stored class path.
        """
        module_depth = (
            1 if self.kind in ('function', 'class') 
            else 2 if self.kind == 'static_method' 
            else None
        )
        return load_from_path(self.path, module_depth=module_depth)
    
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
            raise ValueError(
                "Unrecognized value for self.recovery_mode: "
                f"'{self.recovery_mode}'. Must be 'call' or 'get_callable'."
            )
        
    def manually_recover(self, **kwargs):
        """ 
        """
        self.locked = False
        return self.recover(**kwargs)


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
class TorchDeviceConfig(ArgsConfig):
    device: str = 'cpu'


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
        Convenience method to instantiate a TensorConfig object from a tensor.
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
        if 'dtype' not in shallow_asdict(self.args_cfg):
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
    seed_cfg_list: List[Dict[str, SeedConfig]]
    torch_determinism_cfg_dict : Dict[str, TorchDeterminismConfig]