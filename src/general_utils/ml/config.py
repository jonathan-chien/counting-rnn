from dataclasses import dataclass
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

from ..config.types import ArgsConfig, ContainerConfig, CallableConfig, SeedConfig
if TYPE_CHECKING: from .training import StoppingStrategy

# ---------------------------- Training/Eval -------------------------------- #
@dataclass
class LossTermConfig(ArgsConfig):
    name: str
    loss_fn: Callable[..., torch.Tensor]
    mode: str
    weight: float = 1.
    optimizer: Optional[CallableConfig] = None


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


# ------------------------------ Training ----------------------------------- #
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
class NoImprovementStoppingConfig(ArgsConfig):
    patience: int
    mode: str 
    tol: float = 1e-6


@dataclass
class EarlyStoppingConfig(ArgsConfig):
    metric_name: str
    # patience: int
    # mode: str
    # tol: float
    strategy: 'StoppingStrategy'
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
class RequiresGradConfig(ArgsConfig):
    networks: Dict[str, List[str]]
    mode: Literal['inclusion', 'exclusion']
    requires_grad: bool
    verbose: bool
    description: Optional[str] = None


# ---------------------------- Reproducibility ------------------------------ #
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
     