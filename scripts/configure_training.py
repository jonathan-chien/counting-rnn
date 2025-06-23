from dataclasses import dataclass
import inspect
import os
from typing import Any, Callable, List, Optional, Tuple

import torch

from data.sequences import Sequences
from engine.utils import Logger
from engine.train import EarlyStopping, MetricTracker
from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
from general_utils.config_types import ArgsConfig, ContainerConfig, FactoryConfig
from general_utils.serialization import recursive_instantiation


@dataclass
class LossTermConfig(ArgsConfig):
    name: str
    loss_fn: Callable[..., torch.Tensor]
    mode: str
    weight: float = 1.
    optimizer: Optional[FactoryConfig] = None

@dataclass
class AdamWConfig:
    lr: float =0.001, 
    betas: tuple = (0.9, 0.999), 
    eps: float = 1e-08, 
    weight_decay: float = 0.01, 
    amsgrad: bool =False

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
    print_flush_epoch: bool = False
    print_flush_batch: bool = False

@dataclass
class DataLoaderConfig(ArgsConfig):
    collate_fn: Optional[Callable[[List[Tuple]], Any]]
    batch_size: int = 128
    shuffle: bool = True

@dataclass
class TrainConfig(ContainerConfig):
    loss_terms: List[FactoryConfig[LossTerm]]
    early_stopping: FactoryConfig[EarlyStopping]
    metric_tracker: FactoryConfig[MetricTracker]
    logger: FactoryConfig[Logger]
    dataloader: FactoryConfig[torch.utils.data.DataLoader]

@dataclass
class ValConfig(ContainerConfig):
    loss_terms: List[FactoryConfig[LossTerm]]
    logger: FactoryConfig[Logger]
    dataloader: FactoryConfig[torch.utils.data.DataLoader]


# TODO: write routines to convert functions to serializable form.

# Training.
loss_term_1 = FactoryConfig.from_class(
    LossTerm,
    LossTermConfig(
        name='cross_entropy',
        loss_fn=wrapped_cross_entropy_loss,
        weight=1.,
        optimizer=FactoryConfig.from_class(
            torch.optim.AdamW,
            AdamWConfig(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0.01, 
                amsgrad=False
            ),
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    )
)

loss_term_2 = FactoryConfig.from_class(
    LossTerm,
    LossTermConfig(
        name='spectral_entropy',
        loss_fn=spectral_entropy,
        weight=1.,
        optimizer=FactoryConfig.from_class(
            torch.optim.AdamW,
            AdamWConfig(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0.01, 
                amsgrad=False
            ),
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    )
)

early_stopping = FactoryConfig.from_class(
    EarlyStopping,
    EarlyStoppingConfig(
        metric_name='val_cross_entropy_loss',
        patience=4,
        mode='min',
        min_epochs_before_stopping=20,
        verbose=True,
        disabled=False
    )
)

metric_tracker = FactoryConfig.from_class(
    MetricTracker,
    MetricTrackerConfig(
        metric_name='val_cross_entropy_loss',
        checkpoint_dir='',
        mode='min',
        frequency='best'
    )
)

logger_train = FactoryConfig.from_class(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='train',
        print_flush_epoch=False,
        print_flush_batch=False
    )
)

loader_train = FactoryConfig.from_class(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    )
)

# Validation.
logger_val = FactoryConfig.from_class(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='val',
        print_flush_epoch=False,
        print_flush_batch=False
    )
)

loader_val = FactoryConfig.from_class(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    )
)






train_cfg = TrainConfig(
    loss_terms=[loss_term_1, loss_term_2],
    early_stopping=early_stopping,
    metric_tracker=metric_tracker,
    logger=logger_train,
    dataloader=loader_train
)

val_cfg = ValConfig(
    loss_terms=[loss_term_1, loss_term_2],
    logger=logger_val,
    dataloader=loader_val
)




# Recursive instantation.
train_cfg = recursive_instantiation(train_cfg)
val_cfg = recursive_instantiation(val_cfg)


# configs/models/000/000/000_000_000.json should point to a dummy model.







# loss_term_1_cfg = LossTermConfig(
#     name='cross_entropy',
#     loss_fn=wrapped_cross_entropy_loss,
#     weight=1.,
#     optimizer=None,
#     mode='train'
# )
# loss_term_2_cfg = LossTermConfig(
#     name='spectral_entropy',
#     loss_fn=spectral_entropy,
#     weight=1.,
#     optimizer=None,
#     mode='train'
# )

# early_stopping_cfg = EarlyStoppingConfig(
#     metric_name='val_cross_entropy_loss',
#     patience=4,
#     mode='min',
#     min_epochs_before_stopping=20,
#     verbose=True,
#     disabled=False
# )

# metric_tracker_cfg = MetricTrackerConfig(
#     metric_name='val_cross_entropy_loss',
#     checkpoint_dir='',
#     mode='min',
#     frequency='best'
# )

# logger_train_cfg = LoggerConfig(
#     log_dir='',
#     log_name='train',
#     print_flush_epoch=False,
#     print_flush_batch=False
# )

# loader_train_cfg = DataLoaderConfig(
#     batch_size=128,
#     shuffle=True,
#     collate_fn=Sequences.pad_collate_fn
# )

# logger_test_cfg = LoggerConfig(
#     log_dir='',
#     log_name='test',
#     print_flush_epoch=False,
#     print_flush_batch=False
# )

# train_cfg = TrainConfig(
#     loss_terms_cfg=
# )



# loader_val_cfg = DataLoaderConfig(
#         batch_size=128,
#         shuffle=True,
#         collate_fn=Sequences.pad_collate_fn
#     )

# logger_val_cfg = LoggerConfig(
#         log_dir='',
#         log_name='val',
#         print_flush_epoch=False,
#         print_flush_batch=False
#     )