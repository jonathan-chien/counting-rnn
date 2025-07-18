from dataclasses import dataclass
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ..config import ArgsConfig, CallableConfig


@dataclass
class LossTermConfig(ArgsConfig):
    name: str
    loss_fn: Callable[..., torch.Tensor]
    mode: str
    weight: float = 1.
    optimizer: Optional[CallableConfig] = None

@dataclass
class AdamConfig(ArgsConfig):
    lr: float = 0.001, 
    betas: tuple = (0.9, 0.999), 
    eps: float = 1e-08, 
    amsgrad: bool = False

@dataclass
class AdamWConfig(ArgsConfig):
    lr: float = 0.001, 
    betas: tuple = (0.9, 0.999), 
    eps: float = 1e-08, 
    weight_decay: float = 0.01, 
    amsgrad: bool = False


@dataclass
class EarlyStoppingConfig(ArgsConfig):
    metric_name: str
    patience: int
    mode: str
    min_epochs_before_stopping: int
    verbose: bool = True
    disabled: bool = False


@dataclass
class MetricTrackerConfig(ArgsConfig):
    metric_name: str
    checkpoint_dir: str
    mode: str
    frequency: str = 'best'


@dataclass
class LoggerConfig(ArgsConfig):
    log_dir: str
    log_name: str
    verbose_batch: bool = False
    verbose_epoch: bool = True
    print_flush_epoch: bool = False
    print_flush_batch: bool = False


@dataclass
class DataLoaderConfig(ArgsConfig):
    collate_fn: Optional[Callable[[List[Tuple]], Any]]
    batch_size: int = 128
    shuffle: bool = True


@dataclass
class RequiresGradConfig(ArgsConfig):
    networks: Dict[str, List[str]]
    mode: Literal['inclusion', 'exclusion']
    requires_grad: bool
    verbose: bool
     