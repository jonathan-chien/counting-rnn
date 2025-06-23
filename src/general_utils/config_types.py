from dataclasses import asdict, dataclass, is_dataclass, fields
from typing import Generic, TypeVar
import warnings

from . serialization import get_cls_path, load_from_path


T = TypeVar('T')


@dataclass
class ArgsConfig:
    """Type 1"""
    pass


@dataclass 
class ContainerConfig:
    """Type 2"""
    pass


@dataclass
class FactoryConfig(Generic[T]):
    """ 
    Utility dataclass for storing a configuration of an object of an arbitrary
    class X within a dataclass config hierarchy. The class path for X is stored
    as a string and constructor args are stored in a dataclass; thus, this
    dataclass can easily be converted to a tagged dict and serialized with
    json, and the object of class X can be instantiated following
    de-serialization. 

    Type 3.
    """
    cls_path : str
    args_cfg : ArgsConfig 
    locked: bool = False
    warn_if_locked: bool = True
    raise_exception_if_locked: bool = False

    @classmethod
    def from_class(
        cls, 
        cls_obj, 
        args_cfg, 
        locked=False, 
        warn_if_locked=True, 
        raise_exception_if_locked=False
    ):
        """ 
        Convenience method allowing easy instantiation of a FactoryConfig 
        object by passing in a reference to class X and an ArgsConfig dataclass 
        containing constructor args for X.
        """
        return cls(
            cls_path=get_cls_path(cls_obj),
            args_cfg=args_cfg,
            locked=locked,
            warn_if_locked=warn_if_locked,
            raise_exception_if_locked=raise_exception_if_locked
        )

    def instantiate(self, **kwargs) -> T:
        """ 
        Instantiate object of class X using previously supplied constructor args.
        """
        if self.locked:
            if self.warn_if_locked and not self.raise_exception_if_locked:
                warnings.warn(
                    "The instantiate method was called on this object, but self.locked=True."
                )
            elif self.raise_exception_if_locked:
                raise RuntimeError(
                    "The instantiate method was called on this object, but self.locked=True."
                )
            return self # y = x.instantiate() results in y == x
         
        if not is_dataclass(self.args_cfg):
            raise TypeError(
                "Expected a dataclass for self.args_cfg but got type " 
                f"{type(self.args_cfg)}."
            )
        
        cls = self.get_class()

        print(f"[DEBUG] Instantiating class from: {self.cls_path}")
        print(f"[DEBUG] args_cfg type: {type(self.args_cfg)}")
        for field in fields(self.args_cfg):
            val = getattr(self.args_cfg, field.name)
            print(f"  - {field.name}: type={type(val)} repr={repr(val)}")

        return cls(**asdict(self.args_cfg), **kwargs)
    
    def get_class(self):
        """ 
        Convenience method to retrieve reference to class X from the stored 
        class path.
        """
        return load_from_path(self.cls_path)