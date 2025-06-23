from dataclasses import dataclass
import inspect
import os
from typing import Any, Callable, List, Optional, Tuple

import torch

from data.sequences import Sequences
from models.networks import AutoRNN
from engine.utils import Logger
from engine.train import EarlyStopping, MetricTracker
from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
from general_utils.config_types import ArgsConfig, ContainerConfig, CallableConfig
from general_utils import fileio as io_utils
from general_utils.serialization import deserialize, recursive_instantiation


@dataclass
class LossTermConfig(ArgsConfig):
    name: str
    loss_fn: Callable[..., torch.Tensor]
    mode: str
    weight: float = 1.
    optimizer: Optional[CallableConfig] = None

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
    loss_terms: List[CallableConfig[LossTerm]]
    early_stopping: CallableConfig[EarlyStopping]
    metric_tracker: CallableConfig[MetricTracker]
    logger: CallableConfig[Logger]
    dataloader: CallableConfig[torch.utils.data.DataLoader]

@dataclass
class ValConfig(ContainerConfig):
    loss_terms: List[CallableConfig[LossTerm]]
    logger: CallableConfig[Logger]
    dataloader: CallableConfig[torch.utils.data.DataLoader]


# TODO: write routines to convert functions to serializable form.

# Training.
loss_term_1 = CallableConfig.from_callable(
    LossTerm,
    LossTermConfig(
        name='cross_entropy',
        loss_fn=wrapped_cross_entropy_loss,
        weight=1.,
        optimizer=CallableConfig.from_callable(
            torch.optim.AdamW,
            AdamWConfig(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0.01, 
                amsgrad=False
            ),
            kind='class',
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    ),
    kind='class'
)

loss_term_2 = CallableConfig.from_callable(
    LossTerm,
    LossTermConfig(
        name='spectral_entropy',
        loss_fn=spectral_entropy,
        weight=1.,
        optimizer=CallableConfig.from_callable(
            torch.optim.AdamW,
            AdamWConfig(
                lr=0.001, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
                weight_decay=0.01, 
                amsgrad=False
            ),
            kind='class',
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    ),
    kind='class'
)

early_stopping = CallableConfig.from_callable(
    EarlyStopping,
    EarlyStoppingConfig(
        metric_name='val_cross_entropy_loss',
        patience=4,
        mode='min',
        min_epochs_before_stopping=20,
        verbose=True,
        disabled=False
    ),
    kind='class'
)

metric_tracker = CallableConfig.from_callable(
    MetricTracker,
    MetricTrackerConfig(
        metric_name='val_cross_entropy_loss',
        checkpoint_dir='',
        mode='min',
        frequency='best'
    ),
    kind='class'
)

logger_train = CallableConfig.from_callable(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='train',
        print_flush_epoch=False,
        print_flush_batch=False
    ),
    kind='class'
)

loader_train = CallableConfig.from_callable(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    ),
    kind='class',
    locked=True,
    warn_if_locked=True,
    raise_exception_if_locked=False
)

# Validation.
logger_val = CallableConfig.from_callable(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='val',
        print_flush_epoch=False,
        print_flush_batch=False
    ),
    kind='class'
)

loader_val = CallableConfig.from_callable(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    ),
    kind='class',
    locked=True,
    warn_if_locked=True,
    raise_exception_if_locked=False
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













# configs/datasets/000/000/000_000_000.json should point to dummy dataset.
data_cfg = deserialize('configs/datasets/000/000/000_000_000.json')
from .configure_dataset import build_hypercube_sequences
sequences = build_hypercube_sequences(data_cfg.sequences_cfg)

# Get embedding dimension and a sequence from sequences_cfg.
from .configure_model import test_model
embedding_dim = data_cfg.sequences_cfg.embedder.args_cfg.ambient_dim
tokens = sequences.special_tokens['switch']['label']
seq, labels, _, _, _ = sequences[0]

device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps:0' if torch.backends.mps.is_available() 
    else 'cpu'
)
    
# configs/models/000/000/000_000_000.json should point to a dummy model.
model_cfg = io_utils.load_from_json('configs/models/000/000/000_000_000.json')

model = test_model(
    embedding_dim=embedding_dim, 
    model_cfg=model_cfg,
    tokens=tokens,
    input_=seq,
    device=device
)










# Prepare arguments to evaluate function.
evaluation_val = {
    'dataloader' : loader_val,
    'switch_label' : sequences['val'].special_tokens['switch']['label'],
    'loss_terms' : loss_terms_val,
    'logger': logger_val,
    'compute_mean_for' : ['cross_entropy_loss', 'accuracy'],
    'log_outputs' : False,
    'criteria' : {'accuracy' : compute_accuracy},
    'h_0' : None,
    'deterministic' : True,
    'device' : device,
    'move_results_to_cpu' : True,
    'verbose' : True
}

# Save config.

    


training = train(
    model,
    loader_train,
    loss_terms=loss_terms_train,
    evaluation=evaluation_val,
    h_0=None,
    logger=logger_train,
    criteria={'accuracy' : compute_accuracy},
    compute_mean_for=['cross_entropy_loss', 'accuracy'],
    save_validation_logger=True,
    metric_tracker=metric_tracker,
    early_stopping=early_stopping,
    num_epochs=3,
    device=device,
    deterministic=True
)