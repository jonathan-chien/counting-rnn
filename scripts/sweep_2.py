import copy
from dataclasses import dataclass, asdict, replace
import itertools
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

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






# --------- Training and Testing --------- #



@dataclass
class TestConfig:
    loss_terms_cfg: List
    logger_cfg: Logger
    dataloader_cfg: DataLoaderConfig



@dataclass
class RunConfig:
    sequences_cfg: Dict[str, SequencesConfig]
    model_cfg: AutoRNNConfig
    train_cfg: TrainConfig
    val_cfg: ValConfig



# --------------------------------------------------------------------------- #
# -------------------------------- Configs ---------------------------------- #
# --------------------------------------------------------------------------- #
device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps:0' if torch.backends.mps.is_available() 
        else 'cpu'
    )

# -------------------------- Auxiliary data config -------------------------- #
hypercube_cfg = HypercubeConfig(
    num_dims=2,
    coords=torch.tensor([0, 1], dtype=torch.int64),
    inclusion_set=torch.tensor(
        [[1, 0], [1, 1]],
        dtype=torch.int8
    ),
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

# ----------------------------- Sequences config ---------------------------- #
sequences_cfg = {}

sequences_cfg['train'] = SequencesConfig(
    num_seq=1024,
    seq_order='permute',
    seq_lengths=seq_lengths,
    elem_cls = Hypercube,
    elem_cfg=hypercube_cfg,
    embedder_cfg=embedder_cfg,
)

sequences_cfg['val'] = copy.deepcopy(sequences_cfg['train'])
sequences_cfg['val'].num_seq = 512

sequences_cfg['test'] = copy.deepcopy(sequences_cfg['train'])
sequences_cfg['test'].num_seq = 512


# ------------------------------ Model config ------------------------------- #
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

rnn_type = 'gru'
if rnn_type == 'gru':
    rnn_cfg = GRUConfig(
        input_size=rnn_input_size,
        hidden_size=20,
        num_layers=1,
        bias=True,
        batch_first=True,
        bidirectional=False,
    )
elif rnn_type == 'rnn':
    rnn_cfg = ElmanConfig(
        input_size=rnn_input_size,
        hidden_size=20,
        num_layers=1,
        bias=True,
        nonlinearity='tanh',
        batch_first=True,
        bidirectional=False,
    )
else:
    ValueError(f"Unrecognized value {rnn_type} for `rnn_type`.")

readout_network_cfg = FCNConfig(
    layer_sizes=[rnn_cfg.hidden_size, 80, 2],
    nonlinearities=[torch.nn.GELU(), torch.nn.Identity()],
    dropouts=[0.5, None]
)

model_cfg = AutoRNNConfig(
    input_network_cfg=input_network_cfg,
    rnn_cfg=rnn_cfg,
    readout_network_cfg=readout_network_cfg
)


# ------------------------------ Train config ------------------------------- #
loss_term_1_cfg = LossTermConfig(
    name='cross_entropy',
    loss_fn=wrapped_cross_entropy_loss,
    weight=1.,
    optimizer=None,
    mode='train'
)
loss_term_2_cfg = LossTermConfig(
    name='spectral_entropy',
    loss_fn=spectral_entropy,
    weight=1.,
    optimizer=None,
    mode='train'
)

early_stopping_cfg = EarlyStoppingConfig(
    metric_name='val_cross_entropy_loss',
    patience=4,
    mode='min',
    min_epochs_before_stopping=20,
    verbose=True,
    disabled=False
)

metric_tracker_cfg = MetricTrackerConfig(
    metric_name='val_cross_entropy_loss',
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
    collate_fn=Sequences.pad_collate_fn
)

logger_test_cfg = LoggerConfig(
    log_dir='',
    log_name='test',
    print_flush_epoch=False,
    print_flush_batch=False
)

train_cfg = TrainConfig(
    loss_terms_cfg=
)



loader_val_cfg = DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=Sequences.pad_collate_fn
    )

logger_val_cfg = LoggerConfig(
        log_dir='',
        log_name='val',
        print_flush_epoch=False,
        print_flush_batch=False
    )




run_cfg = RunConfig(
    sequences_cfg=sequences_cfg,
    model_cfg=model_cfg
)


def build_data_model_and_helpers(cfg: RunConfig, base_dir, run_id, device):
    # Build dataset.
    sequences = {
        key : build_hypercube_sequences(val) for key, val in cfg.sequences_cfg.items()
    }

    # Build model.
    model = build_model(cfg.model_cfg, sequences['train'], device)

    # Build auxiliary objects.
    loss_terms = {}
    logger = {}
    dataloader = {}

    # Training.
    loss_terms['train'] = [LossTerm(**asdict(loss_term_1_cfg)), LossTerm(**asdict(loss_term_2_cfg))]
    logger['train'] = Logger(**asdict(cfg.train_cfg.logger_cfg))
    early_stopping = EarlyStopping(**asdict(early_stopping_cfg))
    metric_tracker = MetricTracker(**asdict(metric_tracker_cfg))
    dataloader['train'] = DataLoader(sequences['train'], **asdict(loader_train_cfg))
    for loss_term in cfg.train_cfg.loss_terms_cfg:
        loss_term.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Validation.
    loss_terms['val'] = copy.deepcopy(loss_terms['train'])
    for loss_term in loss_terms['val']: loss_term.mode = 'eval'
    logger['val'] = Logger(**asdict(cfg.val_cfg.logger_cfg))
    dataloader['val'] = DataLoader(sequences['val'], **asdict(cfg.val_cfg.dataloader_cfg))


    # Testing.
    dataloader['test'] = DataLoader(sequences['test'], **asdict(loader_test_cfg))
    logger['test']
    logger['test'] = Logger(**asdict(cfg.val_cfg.logger_cfg))
    loss_terms_test = copy.deepcopy(loss_terms['train'])
    for loss_term in loss_terms_test: loss_term.mode = 'eval'

    # Save config (tied to actual construction).
    # TODO: add code for this

    return model, sequences, dataloader, loss_terms, logger, metric_tracker, early_stopping






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







loader_test_cfg = DataLoaderConfig(
    batch_size=128,
    shuffle=False,
    collate_fn=sequences['train'].pad_collate_fn
)



evaluation_test = copy.deepcopy(evaluation_val)
evaluation_test.update({
    'dataloader' : loader_test,
    'loss_terms' : loss_terms_test,
    'logger' : logger_test,
})

testing = evaluate(model, **evaluation_test)




