from dataclasses import dataclass, is_dataclass, asdict
from typing import Any

from . import io_utils


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
            cls_path=io_utils.get_path(cls_obj),
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
        return io_utils.load_class_from_path(self.cls_path)