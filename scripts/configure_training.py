import copy

import torch

from data.builder import build_split_sequences, get_tokens
from data.sequences import Sequences
from engine.utils import Logger, compute_accuracy
from engine.train import EarlyStopping, MetricTracker, train
from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
from engine.config import (
    AdamWConfig, DataLoaderConfig, LossTermConfig, EarlyStoppingConfig, 
    MetricTrackerConfig, LoggerConfig, TrainFnConfig, RequiresGradConfig, TrainValConfig
)
from models.builder import test_model
from general_utils.config import CallableConfig, TorchDeviceConfig
from general_utils import fileio as io_utils
from general_utils import serialization as serialization_utils
from general_utils import reproducibility as reproducibility_utils

# @dataclass
# class LossTermConfig(ArgsConfig):
#     name: str
#     loss_fn: Callable[..., torch.Tensor]
#     mode: str
#     weight: float = 1.
#     optimizer: Optional[CallableConfig] = None

# @dataclass
# class AdamWConfig:
#     lr: float =0.001, 
#     betas: tuple = (0.9, 0.999), 
#     eps: float = 1e-08, 
#     weight_decay: float = 0.01, 
#     amsgrad: bool =False

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
# class TrainConfig(ContainerConfig):
#     loss_terms: List[CallableConfig[LossTerm]]
#     early_stopping: CallableConfig[EarlyStopping]
#     metric_tracker: CallableConfig[MetricTracker]
#     logger: CallableConfig[Logger]
#     dataloader: CallableConfig[torch.utils.data.DataLoader]

# @dataclass
# class ValConfig(ContainerConfig):
#     loss_terms: List[CallableConfig[LossTerm]]
#     logger: CallableConfig[Logger]
#     dataloader: CallableConfig[torch.utils.data.DataLoader]

REQUIRES_GRAD_REGISTRY = {
    'no_input_network_freeze_except_for_rnn_input': RequiresGradConfig(
        networks={
            'input_network': [],
            'rnn': ['ih'],
            'output_network': []
        },
        mode='exclusion',
        requires_grad=False
    ),
    'with_input_network_freeze_except_for_input_layer': RequiresGradConfig(
        networks={
            'input_network': ['0.weight', '0.bias'],
            'rnn': [],
            'output_network': []
        },
        mode='exclusion',
        requires_grad=False
    ),
    'none': RequiresGradConfig(
        networks={
            'input_network': [],
            'rnn': [],
            'output_network': []
        },
        mode='inclusion',
        requires_grad=True
    )
}

# TODO: check how device is handled; consider making it more robust

# Training.
loss_term_1 = CallableConfig.from_callable(
    LossTerm,
    LossTermConfig(
        name='cross_entropy',
        loss_fn=CallableConfig.from_callable(
            wrapped_cross_entropy_loss,
            args_cfg=None,
            kind='function',
            recovery_mode='get_callable'
        ),
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
            recovery_mode='call',
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    ),
    kind='class',
    recovery_mode='call'
)

loss_term_2 = CallableConfig.from_callable(
    LossTerm,
    LossTermConfig(
        name='spectral_entropy',
        loss_fn=CallableConfig.from_callable(
            spectral_entropy,
            args_cfg=None,
            kind='function',
            recovery_mode='get_callable'
        ),
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
            recovery_mode='call',
            locked=True,
            warn_if_locked=True,
            raise_exception_if_locked=False
        ),
        mode='train'
    ),
    kind='class',
    recovery_mode='call'
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
    kind='class',
    recovery_mode='call'
)

metric_tracker = CallableConfig.from_callable(
    MetricTracker,
    MetricTrackerConfig(
        metric_name='val_cross_entropy_loss',
        checkpoint_dir='',
        mode='min',
        frequency='best'
    ),
    kind='class',
    recovery_mode='call'
)

logger_train = CallableConfig.from_callable(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='train',
        print_flush_epoch=False,
        print_flush_batch=False
    ),
    kind='class',
    recovery_mode='call'
)

dataloader_train = CallableConfig.from_callable(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    ),
    kind='class',
    recovery_mode='call',
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
    kind='class',
    recovery_mode='call'
)

dataloader_val = CallableConfig.from_callable(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    ),
    kind='class',
    recovery_mode='call',
    locked=True,
    warn_if_locked=True,
    raise_exception_if_locked=False
)


device = CallableConfig.from_callable(
    torch.device,
    TorchDeviceConfig(
        device=(
            'cuda' if torch.cuda.is_available() 
            else 'mps:0' if torch.backends.mps.is_available() 
            else 'cpu'
        )
    ),
    kind='class',
    recovery_mode='call',
)

# device = torch.device(
#     'cuda' if torch.cuda.is_available() 
#     else 'mps:0' if torch.backends.mps.is_available() 
#     else 'cpu'
# )


# Args for the eval.evaluate function called in the train.train function.
evaluation = dict(
    dataloader=dataloader_val,
    switch_label='switch_label___',
    loss_terms=[loss_term_1],
    logger=logger_val,
    log_outputs=False,
    criteria={
        'accuracy' : CallableConfig.from_callable(
            compute_accuracy,
            args_cfg=None,
            kind='function',
            recovery_mode='get_callable',
            locked=False
        )
    },
    compute_mean_for=['cross_entropy_loss', 'accuracy'],
    h_0=None,
    deterministic=True,
    device=device,
    move_results_to_cpu=True,
    verbose=True
)

train_fn_cfg = TrainFnConfig(
    dataloader=dataloader_train,
    loss_terms=[loss_term_1],
    evaluation=evaluation,
    save_validation_logger=True,
    h_0=None,
    logger=logger_train,
    criteria={
        'accuracy' : CallableConfig.from_callable(
            compute_accuracy,
            args_cfg=None,
            kind='function',
            recovery_mode='get_callable',
            locked=False
        )
    },
    compute_mean_for=['cross_entropy_loss', 'accuracy'],
    metric_tracker=metric_tracker,
    early_stopping=early_stopping,
    num_epochs=50,
    device=device,
    deterministic=False
)

train_val_cfg = TrainValConfig(
    train_split_size=5000,
    val_split_size=2500,
    requires_grad_cfg=REQUIRES_GRAD_REGISTRY['none'],
    train_fn_cfg=train_fn_cfg
)

# Recursive instantation.
train_val_cfg = serialization_utils.recursive_recover(train_val_cfg)

# configs/datasets/000/000/000_000_000.json should point to dummy dataset.
data_cfg = serialization_utils.deserialize('configs/datasets/000/000/000_000_000.json')
data_cfg = serialization_utils.recursive_recover(data_cfg)
seed_idx = 0

# sequences = {}

# reproducibility_utils.apply_reproducibility_settings(
#     cfg=data_cfg.reproducibility_cfg,
#     split='train',
#     seed_idx=0
# )
# sequences_cfg_copy_train = copy.deepcopy(data_cfg.sequences_cfg)
# sequences_cfg_copy_train.num_seq = train_val_cfg.train_split_size
# sequences['train'] = build_hypercube_sequences(sequences_cfg_copy_train)

# reproducibility_utils.apply_reproducibility_settings(
#     cfg=data_cfg.reproducibility_cfg,
#     split='val',
#     seed_idx=0
# )
# sequences_cfg_copy_val = copy.deepcopy(data_cfg.sequences_cfg)
# sequences_cfg_copy_val.num_seq = train_val_cfg.val_split_size
# sequences['val'] = build_hypercube_sequences(sequences_cfg_copy_val)
sequences = build_split_sequences(
    data_cfg.sequences_cfg,
    data_cfg.reproducibility_cfg,
    split_names=['train', 'val'],
    split_sizes=[train_val_cfg.train_split_size, train_val_cfg.val_split_size],
    seed_ind=[0, 0]
)

# Get embedding dimension and a sequence from sequences_cfg.
embedding_dim = data_cfg.sequences_cfg.embedder.ambient_dim
tokens = get_tokens(sequences['train'], train_val_cfg.train_fn_cfg.device)
seq, labels, _, _, _ = sequences['train'][0]
seq = seq.to(train_val_cfg.train_fn_cfg.device)

# configs/models/000/000/000_000_000.json should point to a dummy model.
model_cfg = serialization_utils.deserialize('configs/models/000/000/000_000_000.json')
# model_cfg = serialization_utils.recursive_recover(model_cfg)
model = test_model(
    embedding_dim=embedding_dim, 
    model_cfg=model_cfg,
    tokens=tokens,
    input_=seq,
    device=train_val_cfg.train_fn_cfg.device
)

# Add model parameters to optimizers.
# train_val_cfg.train_fn_cfg.loss_terms = [
#     term.optimizer.manually_recover(params=model.parameters())
#     for term in train_val_cfg.train_fn_cfg.loss_terms
# ]
for term in train_val_cfg.train_fn_cfg.loss_terms:
    term.optimizer = term.optimizer.manually_recover(params=model.parameters())

# Add dataset to dataloaders.
train_val_cfg.train_fn_cfg.dataloader = train_val_cfg.train_fn_cfg.dataloader.manually_recover(dataset = sequences['train'])
train_val_cfg.train_fn_cfg.evaluation['dataloader'] = train_val_cfg.train_fn_cfg.evaluation['dataloader'].manually_recover(dataset = sequences['train'])



# For testing, just run a few epochs.
train_fn_cfg_copy = copy.deepcopy(train_val_cfg.train_fn_cfg)

train_fn_cfg_copy.num_epochs = 2

# Replace placeholder.
train_fn_cfg_copy.evaluation['switch_label'] = sequences['train'].special_tokens['switch']['label'].to(train_val_cfg.train_fn_cfg.device)
training = train(model, **serialization_utils.shallow_asdict(train_fn_cfg_copy))







# # Prepare arguments to evaluate function.
# evaluation_val = {
#     'dataloader' : val_cfg.dataloader,
#     'switch_label' : sequences['val'].special_tokens['switch']['label'].to(device),
#     'loss_terms' : val_cfg.loss_terms,
#     'logger': val_cfg.logger,
#     'compute_mean_for' : ['cross_entropy_loss', 'accuracy'],
#     'log_outputs' : False,
#     'criteria' : {'accuracy' : compute_accuracy},
#     'h_0' : None,
#     'deterministic' : True,
#     'device' : device,
#     'move_results_to_cpu' : True,
#     'verbose' : True
# }

# # Save config.

    


# training = train(
#     model,
#     dataloader_train,
#     loss_terms=loss_terms_train,
#     evaluation=evaluation_val,
#     h_0=None,
#     logger=logger_train,
#     criteria={'accuracy' : compute_accuracy},
#     compute_mean_for=['cross_entropy_loss', 'accuracy'],
#     save_validation_logger=True,
#     metric_tracker=metric_tracker,
#     early_stopping=early_stopping,
#     num_epochs=3,
#     device=device,
#     deterministic=True
# )