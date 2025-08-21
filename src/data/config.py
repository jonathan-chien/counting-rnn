from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple

import torch

from . import utils as data_utils
from general_utils import config as config_utils
from general_utils import tensor as tensor_utils


@dataclass
class HypercubeConfig(config_utils.ArgsConfig):
    num_dims: int
    coords: Union[torch.Tensor, config_utils.TensorConfig]
    inclusion_set: Optional[Union[torch.Tensor, config_utils.TensorConfig]] = None
    encoding: Union[torch.Tensor, config_utils.TensorConfig] = torch.tensor([0, 1], dtype=torch.int8)
    vertices_pmfs: Optional[Tuple[torch.Tensor]] = None
    dtype: Union[torch.dtype, str] = 'torch.int8'

@dataclass
class SeqLengths:
    """ 
    Helper class for validating and storing in a format compatible with the
    Sequence class N distributions over the respective lengths of N sequences,
    for a natural number N.

    lengths : dict
        dict where each key is the name of a kind of sequence (e.g. 'pos', 
        'neg'), and each value is a dict with the following keys:
            'support' : 1D tensor of non-negative ints.
            'pmf' : 1D tensor of probability masses, same legnth as 'support'.
    """
    lengths: Dict[str, Dict[str, Union[torch.Tensor, config_utils.TensorConfig]]]

    def validate(self):
        for name, entry in self.lengths.items():
            try:
                support, pmf = entry['support'], entry['pmf']
                tensor_utils.validate_tensor(support, 1)
                data_utils.validate_pmf(pmf, len(support))
            except Exception as e:
                raise ValueError(f"Validation failed for '{name}'.") from e
            

@dataclass
class EmbedderConfig(config_utils.ArgsConfig):
    ambient_dim: int
    mean_center: bool = False
    offset_1: Optional[Union[torch.Tensor, config_utils.TensorConfig]] = None
    offset_2: Optional[Union[torch.Tensor, config_utils.TensorConfig]] = None
    method: Union[str, Union[torch.Tensor, config_utils.TensorConfig]] = 'random_rotation'
    noise_distr: Optional[config_utils.CallableConfig] = None


@dataclass
class SequencesConfig(config_utils.ContainerConfig):
    seq_lengths: SeqLengths
    # elem_cls: type # E.g. Hypercube
    # elem_cfg: Any # E.g. HypercubeConfig
    elem: config_utils.CallableConfig
    embedder: config_utils.CallableConfig
    num_seq: int
    seq_order: str = 'permute'


@dataclass
class NormalDistrConfig(config_utils.ArgsConfig):
    loc : float 
    scale : float 



@dataclass
class SplitConfig(config_utils.ContainerConfig):
    train: int
    val: int
    test: int


@dataclass
class DataConfig(config_utils.ContainerConfig):
    sequences_cfg: Dict[str, SequencesConfig]
    split_cfg: SplitConfig
    


