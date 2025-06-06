import copy
from dataclasses import dataclass, asdict, replace
import itertools
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
if src_path not in sys.path: sys.path.insert(0, src_path)

from data.sequences import Hypercube, Sequences, SeqLengths, Embedder
from data import utils as data_utils
from models.networks import FCN, AutoRNN
from engine.train import EarlyStopping, MetricTracker, train
from engine.utils import Logger
from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
from engine.utils import compute_accuracy
from engine.eval import evaluate

@dataclass
class HypercubeConfig:
    num_dims: int
    coords: torch.Tensor
    inclusion_set: torch.Tensor = None
    encoding: torch.Tensor = torch.tensor([0, 1], dtype=torch.int8)
    
@dataclass
class EmbedderConfig:
    ambient_dim: int
    mean_center: bool = False
    offset_1: Optional[torch.Tensor] = None
    offset_2: Optional[torch.Tensor] = None
    method: Union[str, torch.Tensor] = 'random_rotation'
    noise_distr: Optional[torch.distributions.Distribution] = None

@dataclass
class SequencesConfig:
    num_seq: int
    seq_order: str = 'permute'
    seq_lengths: SeqLengths
    elem_cls: Hypercube
    elem_config: HypercubeConfig
    embedder_config: EmbedderConfig

def build_hypercube_sequences(cfg: SequencesConfig) -> Sequences:
    hypercube = cfg.elem_cls(**asdict(cfg.elem_cfg))
    embedder = Embedder(**asdict(cfg.embedder_cfg))

    # Check that number of variables is large enough.
    if embedder.ambient_dim < hypercube.num_dims + 3: 
        raise ValueError(
            "`embedder.ambient_dim` must be at least 3 greater than hypercube " 
            f"dimensionality {hypercube.num_dims}, but got {embedder.ambient_dim}."
        )
    
    return Sequences(
        num_seq=cfg.num_seq,
        len_distr=cfg.seq_lengths.lengths,
        elem_distr=hypercube.vertices,
        transform=embedder,
        seq_order=cfg.seq_order
    )

@dataclass
class FCNConfig:
    layer_sizes: Optional[List[int]]
    nonlinearities: List[Optional[torch.nn.Module]]
    dropouts: List[Optional[torch.nn.Module]]

@dataclass
class BaseRNNConfig:
    input_size: int
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    batch_first: bool = True
    dropout: float = 0
    bidirectional: bool = False

    def build(self) -> torch.nn.modules.rnn.RNNBase:
        raise NotImplementedError()
    
@dataclass
class ElmanConfig(BaseRNNConfig):
    nonlinearity: str = 'tanh'
    def build(self):
        return torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity=self.nonlinearity,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
    
@dataclass
class GRUConfig(BaseRNNConfig):
    def build(self):
        return torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

@dataclass
class AutoRNNConfig:
    input_network_cfg: FCNConfig
    rnn_cfg: BaseRNNConfig
    readout_network_cfg: FCNConfig

def build_model(cfg: AutoRNNConfig, sequences, device) -> AutoRNN:
    input_network = FCN(**asdict(cfg.input_network_config))
    rnn = cfg.rnn_config.build()
    readout_network = FCN(**asdict(cfg.readout_network_config))

    tokens = sequences.transform(
        torch.cat(
            (sequences.special_tokens['count']['token'].unsqueeze(0), 
            sequences.special_tokens['eos']['token'].unsqueeze(0)), 
            dim=0
        )
    ).to(device)

    return AutoRNN(input_network, rnn, readout_network, tokens)


# --------- Training and Testing --------- #

@dataclass
class LossTermConfig:
    name: str
    loss_fn: Callable[..., torch.Tensor]
    weight: float = 1.
    optimizer: Optional[torch.optim.Optimizer] = None
    mode: str

@dataclass
class EarlyStoppingConfig:
    metric_name: str
    patience: int
    mode: str
    min_epochs_before_stopping: int
    verbose: bool = True
    disabled: bool = False

@dataclass
class MetricTrackerConfig:
    metric_name: str
    checkpoint_dir: str
    frequency: str = 'best'
    mode: str

@dataclass
class LoggerConfig:
    log_dir: str
    log_name: str
    print_flush_epoch: bool = False
    print_flush_batch: bool = False

@dataclass
class DataLoaderConfig:
    batch_size: int = 128
    shuffle: bool = True
    collage_fn: Optional[Callable[[List[Tuple]], Any]]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Build dataset.
hypercube_cfg = HypercubeConfig(
    num_dims=2,
    coords=torch.tensor([0]),
    inclusion_set=torch.tensor([1]),
    encoding=torch.tensor([0, 1], dtype=torch.int8)
)

seq_lengths = SeqLengths(
    lengths={
        'pos' : {
            'support' : torch.arange(0, 10),
            'pmf' : data_utils.uniform_pmf(10)
        },
        'neg' : {
            'support' : torch.arange(0, 5),
            'pmf' : data_utils.uniform_pmf(5)
        }
    }
)

embedder_cfg = EmbedderConfig(
    ambient_dim=5,
    mean_center=False,
    offset_1=-torch.tile(torch.tensor([0.5]), (hypercube_cfg.num_dims + 3,)), # Plus 3 for the dimensions corresponding to special tokens
    offset_2=None,
    method='random_rotation',
    noise_distr=torch.distributions.Normal(0, 0.05)
)


sequences_cfg = dict()

sequences_cfg['train'] = SequencesConfig(
    num_seq=1024,
    num_vars=5,
    seq_order='permute',
    seq_lengths=seq_lengths,
    hypercube_cfg=hypercube_cfg,
    embedder_cfg=embedder_cfg,
)

sequences_cfg['val'] = copy.deepcopy(sequences_cfg['train'])
sequences_cfg['val'].num_seq = 512

sequences_cfg['test'] = copy.deepcopy(sequences_cfg['train'])
sequences_cfg['test'].num_seq = 512

sequences = {
    key : build_hypercube_sequences(val) for key, val in sequences_cfg.items()
}



device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps:0' if torch.backends.mps.is_available() 
    else 'cpu'
)

input_network_cfg = FCNConfig(
    layer_sizes=[embedder_cfg.ambient_dim, 50],
    nonlinearities=[torch.nn.ReLU()],
    dropouts=[None]
)

rnn_input_size = (
    input_network_cfg.layer_sizes[-1] 
    if input_network_cfg.layer_sizes is not None 
    else embedder_cfg.ambient_dim
)
rnn_cfg = GRUConfig(
    input_size= rnn_input_size,
    hidden_size=20,
    num_layers=1,
    bias=True,
    batch_first=True,
    bidirectional=False,
    device=None
)

readout_network_cfg = FCNConfig(
    layer_sizes=[rnn_cfg.hidden_size, 80, 2],
    nonlinearities=[torch.nn.GELU(), None],
    dropouts=[torch.nn.Dropout(p=0.5), None]
)

model_cfg = AutoRNNConfig(
    input_network_cfg=input_network_cfg,
    rnn_cfg=rnn_cfg,
    readout_network_cfg=readout_network_cfg
)

model = build_model(model_cfg, sequences['train'], device)










loss_term_1_cfg = LossTermConfig(
    name='cross_entropy',
    loss_fn=wrapped_cross_entropy_loss,
    weight=1.,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
    mode='train'
)
loss_term_2_cfg = LossTermConfig(
    name='spectral_entropy',
    loss_fn=spectral_entropy,
    weight=1.,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    mode='train'
)

early_stopping_cfg = EarlyStoppingConfig(
    metric_name='val_loss',
    patience=4,
    mode='min',
    min_epochs_before_stopping=20,
    verbose=True,
    disabled=False
)

metric_tracker_cfg = MetricTrackerConfig(
    metric_name='val_loss',
    checkpoint_dir='',
    mode='min',
    frequency='best'
)

logger_train_cfg = LoggerConfig(
    log_dir='',
    log_name='train',
    print_flush_epoch=False,
    print_flush_batch=False
)

loader_train_cfg = DataLoaderConfig(
    batch_size=128,
    shuffle=True,
    collage_fn=sequences['train'].pad_collate_fn
)

loss_terms_train = [LossTerm(**asdict(loss_term_1_cfg)), LossTerm(**asdict(loss_term_2_cfg))]
early_stopping = EarlyStopping(**asdict(early_stopping_cfg))
metric_tracker = MetricTracker(**asdict(metric_tracker_cfg))
logger_train = Logger(**asdict(logger_train_cfg))
loader_train = DataLoader(**asdict(loader_train_cfg))




loader_val_cfg = DataLoaderConfig(
    batch_size=128,
    shuffle=True,
    collage_fn=sequences['train'].pad_collate_fn
)
loss_terms_val = copy.deepcopy(loss_terms_train)
for loss_term in loss_terms_val: loss_term.mode = 'eval'

logger_val_cfg = LoggerConfig(
    log_dir='',
    log_name='val',
    print_flush_epoch=False,
    print_flush_batch=False
)


logger_val = Logger(**asdict(logger_val_cfg))
loader_val = DataLoader(**asdict(loader_val_cfg))



# Prepare arguments to evaluate function.
evaluation_val = {
    'data_loader' : loader_val,
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
    num_epochs=150,
    device=device,
    deterministic=True
)







loader_test_cfg = DataLoaderConfig(
    batch_size=128,
    shuffle=False,
    collage_fn=sequences['train'].pad_collate_fn
)

loader_test = DataLoader(**asdict(loader_test_cfg))

logger_test_cfg = LoggerConfig(
    log_dir='',
    log_name='test',
    print_flush_epoch=False,
    print_flush_batch=False
)


logger_test = Logger(**asdict(logger_test_cfg))


loss_terms_test = copy.deepcopy(loss_terms_train)
for loss_term in loss_terms_test: loss_term.mode = 'eval'

evaluation_test = copy.deecopy(evaluation_val)
evaluation_test.update({
    'dataloader' : loader_test,
    'loss_terms' : loss_terms_test,
    'logger' : logger_test,
})

testing = evaluate(model, **evaluation_test)


