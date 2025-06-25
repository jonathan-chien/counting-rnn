from dataclasses import dataclass
from typing import List, Optional

from general_utils.config import ArgsConfig, CallableConfig


@dataclass
class ReLUConfig(ArgsConfig):
    inplace: bool = False


@dataclass
class GELUConfig(ArgsConfig):
    approximate: str = 'none'


@dataclass
class IdentityConfig(ArgsConfig):
    pass


@dataclass
class FCNConfig(ArgsConfig):
    layer_sizes: Optional[List[int]]
    nonlinearities: List[Optional[CallableConfig]]
    dropouts: List[Optional[CallableConfig]]


@dataclass
class BaseRNNConfig(ArgsConfig):
    input_size: int
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    batch_first: bool = True
    dropout: float = 0
    bidirectional: bool = False
    

@dataclass
class ElmanConfig(BaseRNNConfig):
    nonlinearity: str = 'tanh'
    

@dataclass
class GRUConfig(BaseRNNConfig):
    pass


@dataclass
class AutoRNNConfig(ArgsConfig):
    input_network: CallableConfig
    rnn: CallableConfig
    readout_network: CallableConfig