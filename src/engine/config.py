from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from general_utils.config.types import ArgsConfig, ContainerConfig, CallableConfig, TensorConfig
from general_utils.ml.config import RequiresGradConfig


@dataclass
class EvalFnConfig(ArgsConfig):
    """ 
    Args to evaluate function, called within the train function. Stored under 
    the 'evaluation' field of TrainConfig.
    """
    loss_terms: List[CallableConfig]
    dataloader: CallableConfig
    switch_label: Any # Needs to be torch tensor, but could probably be int too
    log_outputs: bool
    criteria: Dict[str, CallableConfig]
    deterministic: bool
    device: str
    move_results_to_cpu: bool
    logger: Optional[CallableConfig] = None
    compute_mean_for: Optional[List[str]] = None
    h_0: Optional[Union[Callable, TensorConfig]] = None


@dataclass
class TrainFnConfig(ArgsConfig):
    """ 
    """
    dataloader: CallableConfig
    loss_terms: List[CallableConfig]
    evaluation: EvalFnConfig
    # save_validation_logger: bool
    criteria: Dict[str, CallableConfig]
    num_epochs: int
    device: str
    deterministic: bool
    h_0: Optional[Union[Callable, TensorConfig]] = None
    logger_train: Optional[CallableConfig] = None
    compute_mean_for: Optional[List[str]] = None
    metric_tracker: Optional[CallableConfig] = None
    early_stopping: Optional[CallableConfig] = None


@dataclass
class TrainingConfig(ContainerConfig):
    train_fn_cfg: TrainFnConfig
    requires_grad_cfg: RequiresGradConfig
    train_split_seed_idx: int = 0
    val_split_seed_idx: int = 0
    

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

