from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from general_utils.config import ArgsConfig, ContainerConfig, CallableConfig, TensorConfig
from general_utils.ml.config import RequiresGradConfig

# @dataclass
# class LossTermConfig(ArgsConfig):
#     name: str
#     loss_fn: Callable[..., torch.Tensor]
#     mode: str
#     weight: float = 1.
#     optimizer: Optional[CallableConfig] = None


# @dataclass
# class AdamWConfig(ArgsConfig):
#     lr: float = 0.001, 
#     betas: tuple = (0.9, 0.999), 
#     eps: float = 1e-08, 
#     weight_decay: float = 0.01, 
#     amsgrad: bool = False


# @dataclass
# class EarlyStoppingConfig(ArgsConfig):
#     metric_name: str
#     patience: int
#     mode: str
#     min_epochs_before_stopping: int
#     verbose: bool = True
#     disabled: bool = False


# @dataclass
# class MetricTrackerConfig(ArgsConfig):
#     metric_name: str
#     checkpoint_dir: str
#     mode: str
#     frequency: str = 'best'


# @dataclass
# class LoggerConfig(ArgsConfig):
#     log_dir: str
#     log_name: str
#     print_flush_epoch: bool = False
#     print_flush_batch: bool = False


# @dataclass
# class DataLoaderConfig(ArgsConfig):
#     collate_fn: Optional[Callable[[List[Tuple]], Any]]
#     batch_size: int = 128
#     shuffle: bool = True

# @dataclass
# class TrainConfig(ArgsConfig):
#     loss_terms: List[CallableConfig[LossTerm]]
#     early_stopping: CallableConfig[EarlyStopping]
#     metric_tracker: CallableConfig[MetricTracker]
#     logger: CallableConfig[Logger]
#     dataloader: CallableConfig[torch.utils.data.DataLoader]


# @dataclass
# class RequiresGradConfig(ArgsConfig):
#     networks: Dict[str, List[str]]
#     mode: str
#     requires_grad: bool


@dataclass
class EvalFnConfig(ArgsConfig):
    """ 
    Args to evaluate function, called within the train function. Stored under 
    the 'evaluation' field of TrainConfig.
    """
    loss_terms: List[CallableConfig]
    logger: Optional[CallableConfig]
    dataloader: CallableConfig
    switch_label: Any # Needs to be torch tensor, but could probably be int too
    log_outputs: bool
    criteria: Dict[str, CallableConfig]
    compute_mean_for: Optional[List[str]]
    h_0: Optional[Union[Callable, TensorConfig]]
    deterministic: bool
    device: str
    move_results_to_cpu: bool


@dataclass
class TrainFnConfig(ArgsConfig):
    """ 
    """
    dataloader: CallableConfig
    loss_terms: List[CallableConfig]
    evaluation: EvalFnConfig
    # save_validation_logger: bool
    h_0: Optional[Union[Callable, TensorConfig]]
    logger_train: Optional[CallableConfig]
    criteria: Dict[str, CallableConfig]
    compute_mean_for: Optional[List[str]]
    metric_tracker: Optional[CallableConfig]
    early_stopping: Optional[CallableConfig]
    num_epochs: int
    device: str
    deterministic: bool


# @dataclass
# class SplitConfig(ArgsConfig):
#     split_name: str
#     split_size: int
#     seed_idx: int


@dataclass
class TrainingConfig(ContainerConfig):
    train_fn_cfg: TrainFnConfig
    train_split_seed_idx: 0
    val_split_seed_idx: 0
    requires_grad_cfg: RequiresGradConfig
    

@dataclass
class TestingConfig(ContainerConfig):
    eval_fn_cfg: EvalFnConfig
    test_split_seed_idx: 0
    

@dataclass
class ExperimentConfig(ContainerConfig):
    """
    """
    data_training_cfg: Optional[Union[ArgsConfig, ContainerConfig]] = None 
    training_cfg: Optional[Union[ArgsConfig, ContainerConfig]] = None 
    model_cfg: Optional[Union[ArgsConfig, ContainerConfig]] = None 
    data_test_cfg: Optional[Union[ArgsConfig, ContainerConfig]] = None 
    testing_cfg: Optional[Union[ArgsConfig, ContainerConfig]] = None 

# @dataclass
# class TrainConfig(ArgsConfig):
#     loss_terms: List[CallableConfig[LossTerm]]
#     early_stopping: CallableConfig[EarlyStopping]
#     metric_tracker: CallableConfig[MetricTracker]
#     logger: CallableConfig[Logger]
#     dataloader: CallableConfig[torch.utils.data.DataLoader]

