from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from general_utils.config import ArgsConfig, ContainerConfig, CallableConfig, TensorConfig
from general_utils.ml.config import RequiresGradConfig


@dataclass
class EvalFnConfig(ArgsConfig):
    """ 
    Args to evaluate function, called within the train function. Stored under 
    the 'evaluation' field of TrainConfig.
    """
    loss_terms: List[CallableConfig]
    logger: Optional[CallableConfig] = None
    dataloader: CallableConfig
    switch_label: Any # Needs to be torch tensor, but could probably be int too
    log_outputs: bool
    criteria: Dict[str, CallableConfig]
    compute_mean_for: Optional[List[str]] = None
    h_0: Optional[Union[Callable, TensorConfig]] = None
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
    h_0: Optional[Union[Callable, TensorConfig]] = None
    logger_train: Optional[CallableConfig] = None
    criteria: Dict[str, CallableConfig]
    compute_mean_for: Optional[List[str]] = None
    metric_tracker: Optional[CallableConfig] = None
    early_stopping: Optional[CallableConfig] = None
    num_epochs: int
    device: str
    deterministic: bool


@dataclass
class TrainingConfig(ContainerConfig):
    train_fn_cfg: TrainFnConfig
    train_split_seed_idx: int = 0
    val_split_seed_idx: int = 0
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

