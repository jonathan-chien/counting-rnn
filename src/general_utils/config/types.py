from dataclasses import dataclass, fields, replace
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin
import warnings
warnings.simplefilter("always")

import torch

from . import serialization 


# ---------------------------- Dataclass configs ---------------------------- #
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
    # warn_if_locked: bool = True
    # raise_exception_if_locked: bool = False
    if_recover_while_locked: Literal['print', 'warn', 'raise_exception', 'silent'] = 'print'

    @classmethod
    def from_callable(
        cls, 
        callable_: Any, 
        args_cfg: Optional[ArgsConfig], 
        kind: str,
        recovery_mode: str,
        *,
        locked: bool = False, 
        if_recover_while_locked: str = 'print'
    ):
        """ 
        """
        return cls(
            path=cls._get_path(callable_, kind),
            args_cfg=args_cfg,
            kind=kind,
            recovery_mode=recovery_mode,
            locked=locked,
            if_recover_while_locked=if_recover_while_locked
        )
    
    @staticmethod
    def _get_path(callable_, kind):
        """ 
        """
        if kind == 'class':
            return serialization.get_cls_path(callable_)
        elif kind in ('function', 'static_method'):
            return serialization.get_fn_path(callable_)
        else:
            raise ValueError(
                f"Unrecognized value {kind} for kind. Must be 'class' or 'function'."
            )
    
    def _get_args_cfg_with_dtype(self):
        """ 
        Return updated copy of self.args_cfg with torch.dtype string placeholders
        replaced by actual torch.dtypes.
        """
        def is_dtype_annotation(annotation):
            """
            Check if current dataclass type annotation is torch,dtype or 
            Union[torch.dtype, ...].
            """
            # Return true if annotation is just torch.dtype.
            if annotation is torch.dtype:
                return True
            
            # Else, return true if is Union and any element is torch.dtype.
            origin = get_origin(annotation)
            if origin is Union: # Optional[x] ~ Union[x, None]
                return any(elem is torch.dtype for elem in get_args(annotation))
            
            return False
        
        def get_dtype(s):
            """
            Resolve a torch.dtype placeholder string, e.g. 'torch.float32' into
            the actual torch.dtype, e.g. torch.float32.
            """
            dtype_name = s.split('.')[-1]
            try:
                dtype = getattr(torch, dtype_name)
            except AttributeError:
                raise ValueError(f"Invalid torch.dtype string: {s}.")

            if isinstance(dtype, torch.dtype):
                return dtype
            else:
                raise ValueError(
                    f"torch.dtype string {s} could not be resolved to a valid torch.dtype."
                )
            
        updated = {}
        for f in fields(self.args_cfg):
            value = getattr(self.args_cfg, f.name)
            if is_dtype_annotation(f.type) and isinstance(value, str):
                updated[f.name] = get_dtype(value)
        
        return replace(self.args_cfg, **updated)
        
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
        args_cfg_with_dtype = self._get_args_cfg_with_dtype()
        return callable_(**{**serialization.shallow_asdict(args_cfg_with_dtype), **kwargs})
    
    def _get_callable(self):
        """ 
        Retrieve reference to callable from the stored class path.
        """
        module_depth = (
            1 if self.kind in ('function', 'class') 
            else 2 if self.kind == 'static_method' 
            else None
        )
        return serialization.load_from_path(self.path, module_depth=module_depth)
    
    def _output_if_locked(self):
        """ 
        """
        message = (
            "The `recover` method was called on an object with self.path=" 
            f"{self.path}, but recovery will be deferred, as self.locked=True."
        )

        if self.if_recover_while_locked == 'print':
            print(message)
        elif self.if_recover_while_locked == 'warn':
            warnings.warn(message)
        elif self.if_recover_while_locked == 'raise_exception':
            raise RuntimeError(message)
        elif self.if_recover_while_locked != 'silent':
            raise ValueError(
                f"Unrecognized value {self.if_recover_while_locked} for "
                "self.if_recover_while_locked. Must be one of ['warn', 'raise_exception', 'silent']."
            )

    def recover(self, **kwargs):
        """ 
        """
        if self.locked:
            self._output_if_locked()
            return self
        elif self.recovery_mode == 'call':
            # print(self.path) # Development/debugging
            return self._call(**kwargs)
        elif self.recovery_mode == 'get_callable':
            if len(kwargs) != 0:
                raise RuntimeError(
                    "It seems kwargs were passed to the `recover` method. Since " 
                    "self.recovery_mode='get_callable', `recover` is a wrapper " 
                    "to `_get_callable`, which takes no arguments."
                )
            # print(self.path) # Development/debugging
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
    dtype: Union[torch.dtype, str]
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
        if_recover_while_locked: str = 'print'
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
            if_recover_while_locked=if_recover_while_locked
        )
    
    # def recover(self):
    #     # Key 'dtype' points to a string, must convert to torch.dtype.
    #     if 'dtype' not in serialization.shallow_asdict(self.args_cfg):
    #         raise KeyError(
    #             "TensorConfig object must contain a dtype attribute for " 
    #             "accurate reconstruction."
    #         )
    #     try:
    #         dtype = getattr(torch, self.args_cfg.dtype.rsplit('.', 1)[-1])
    #         if not isinstance(dtype, torch.dtype):
    #             raise TypeError
    #     except (AttributeError, TypeError):
    #         raise ValueError(
    #             f"self.args_cfg.dtype = {self.args_cfg.dtype} is invalid. Must "
    #             "be string equivalent of valid torch.dtype, e.g. 'torch.float32."
    #         )
        
    #     updated_args_cfg = replace(self.args_cfg, dtype=dtype)
    #     temporary_tensor_config = type(self).from_callable(
    #         callable_=torch.tensor,
    #         args_cfg=updated_args_cfg,
    #         kind=self.kind,
    #         recovery_mode=self.recovery_mode,
    #         locked=self.locked,
    #         # warn_if_locked=self.warn_if_locked,
    #         # raise_exception_if_locked=self.raise_exception_if_locked
    #         if_recover_while_locked=self.if_recover_while_locked
    #     )
    #     return super(TensorConfig, temporary_tensor_config).recover()
    # def recover(self, dtype_str='raise'):
    #     """
    #     Before 2025-08-20, this class' recover method was responsible for
    #     converting string placeholders for torch.dtypes to actual torch.dtypes
    #     and then calling the parent class' recover method. Conversion of
    #     torch.dtype placeholder strings to torch.dtypes has now been centralized
    #     in the recursive_recover function. As such, this class' recover method
    #     will now only raise if torch.dtype placeholder strings have not been
    #     converted at the time of a call to this method; it will not silently
    #     convert strings to torch.dtypes, though optional automatic conversion
    #     preceded by a warning is supported.
    #     """
    #     if not isinstance(self.args_cfg, TensorArgsConfig):
    #         raise TypeError(
    #             f"self.args_cfg must be an instance of TensorArgsConfig but got type {type(self.args_cfg)}."
    #         )
    #     if not any(f.name == 'dtype' for f in fields(self.args_cfg)):
    #         raise AttributeError(
    #             "TensorConfig object must contain a dtype attribute for " 
    #             "accurate reconstruction."
    #         )
    #     if isinstance(self.args_cfg.dtype, str):
    #         if dtype_str == 'raise':
    #             raise RuntimeError(
    #                 "TensorConfig.recover() called while self.args_cfg.dtype " 
    #                 f"is still a string '{self.args_cfg.dtype}'; conversion to torch.dtype "
    #                 "should occur in serialization.recursive_recover (2025-08-20)." 
    #                 "Make sure that TensorArgsConfig's dtype attribute is typed as Union[torch.dtype, str]."
    #             )
    #         elif dtype_str == 'convert':
    #             warnings.warn(
    #                 f"A string {self.args_cfg.dtype} was detected for self.args_cfg.dtype. "
    #                 "Conversion to torch.dtype will be attempted without mutating " 
    #                 "self.args_cfg, followed by a call to CallableConfig.recover(). "
    #                 "No subsequent exception should be interpreted to mean success."
    #             )
    #             updated_args_cfg = replace(
    #                 self.args_cfg, 
    #                 dtype=serialization.get_torch_dtype_from_str(self.args_cfg.dtype)
    #             )
    #             # temporary_tensor_config = type(self).from_callable(
    #             #     callable_=torch.tensor,
    #             #     args_cfg=updated_args_cfg,
    #             #     kind=self.kind,
    #             #     recovery_mode=self.recovery_mode,
    #             #     locked=self.locked,
    #             #     if_recover_while_locked=self.if_recover_while_locked
    #             # )
    #             temporary_tensor_config = replace(self, args_cfg=updated_args_cfg)
    #             # return super(TensorConfig, temporary_tensor_config).recover()
    #             return CallableConfig.recover(temporary_tensor_config)
    #         else:
    #             raise ValueError(
    #                 f"Unrecognized value {dtype_str} for dtype_str. Must be 'raise' or 'convert'."
    #             )

    #     return super().recover()
    

# ----------------------------- Seed utils ---------------------------------- #
@dataclass
class SeedConfig(ArgsConfig):
    torch_seed: int
    cuda_seed: int