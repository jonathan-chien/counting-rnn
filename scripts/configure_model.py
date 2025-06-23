from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

from models.networks import FCN, AutoRNN
from general_utils.config_types import ArgsConfig, ContainerConfig, FactoryConfig
from general_utils.serialization import serialize_and_deserialize, recursive_instantiation
from general_utils import fileio as io_utils

@dataclass
class ReLUConfig(ArgsConfig):
    inplace: bool = False


@dataclass
class GELUConfig(ArgsConfig):
    approximate: str = 'none'


@dataclass
class IdentityConfig(ArgsConfig):
    pass

# @dataclass
# class FCNConfig:
#     layer_sizes: Optional[List[int]]
#     nonlinearities: List[Optional[FactoryConfig]]
#     dropouts: List[Optional[FactoryConfig]]

# @dataclass
# class BaseRNNConfig:
#     input_size: int
#     hidden_size: int
#     num_layers: int = 1
#     bias: bool = True
#     batch_first: bool = True
#     dropout: float = 0
#     bidirectional: bool = False

#     def build(self) -> torch.nn.modules.rnn.RNNBase:
#         raise NotImplementedError()
    
# @dataclass
# class ElmanConfig(BaseRNNConfig):
#     nonlinearity: str = 'tanh'
#     def build(self):
#         return torch.nn.RNN(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             nonlinearity=self.nonlinearity,
#             bias=self.bias,
#             batch_first=self.batch_first,
#             dropout=self.dropout,
#             bidirectional=self.bidirectional
#         )
    
# @dataclass
# class GRUConfig(BaseRNNConfig):
#     def build(self):
#         return torch.nn.GRU(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             bias=self.bias,
#             batch_first=self.batch_first,
#             dropout=self.dropout,
#             bidirectional=self.bidirectional
#         )

# @dataclass
# class AutoRNNConfig:
#     input_network_cfg: FCNConfig
#     rnn_cfg: BaseRNNConfig
#     readout_network_cfg: FCNConfig

@dataclass
class FCNConfig(ArgsConfig):
    layer_sizes: Optional[List[int]]
    nonlinearities: List[Optional[FactoryConfig]]
    dropouts: List[Optional[FactoryConfig]]

@dataclass
class BaseRNNConfig(ArgsConfig):
    input_size: int
    hidden_size: int
    num_layers: int = 1
    bias: bool = True
    batch_first: bool = True
    dropout: float = 0
    bidirectional: bool = False
    
@dataclass
class ElmanConfig(BaseRNNConfig):
    nonlinearity: str = 'tanh'
    
@dataclass
class GRUConfig(BaseRNNConfig):
    pass

@dataclass
class AutoRNNConfig(ArgsConfig):
    input_network: FactoryConfig
    rnn: FactoryConfig
    readout_network: FactoryConfig

def build_model(cfg: AutoRNNConfig, tokens) -> AutoRNN:
    # input_network = cfg.input_network_cfg.instantiate()
    # rnn = cfg.rnn_cfg.instantiate()
    # readout_network = cfg.readout_network_cfg.instantiate()
    # return AutoRNN(input_network, rnn, readout_network, tokens)
    components = {
        name : network.instantiate() for name, network in asdict(cfg).items()
    }
    return AutoRNN(**components, tokens=tokens)

def get_tokens(sequences, device):
    return sequences.transform(
        torch.cat(
            (
                sequences.special_tokens['count']['token'].unsqueeze(0), 
                sequences.special_tokens['eos']['token'].unsqueeze(0)
            ), 
            dim=0
        )
    ).to(device)


# ------------------------------ Set directory ------------------------------ #
output_dir, filename = io_utils.make_file_dir_and_id(
    base_dir='configs/models',
    sub_dir_1='000',
    sub_dir_2='000',
    file_ind = ('000', '000', '000')
)

# TODO: remove _cfg suffix or consider _factory, breaking convention only because this is a standalone variable (but probably better not to do this)
input_network_cfg = FactoryConfig.from_class(
    FCN, 
    FCNConfig(
        layer_sizes=['embedding_dim', 50],
        nonlinearities=[FactoryConfig.from_class(torch.nn.ReLU, ReLUConfig())],
        dropouts=[None]
    )
)

rnn_input_size = (
    input_network_cfg.args_cfg.layer_sizes[-1] 
    if input_network_cfg.args_cfg.layer_sizes is not None 
    else 'embedding_dim'
)

rnn_type = 'gru'
if rnn_type == 'gru':
    rnn_cfg = FactoryConfig.from_class(
        torch.nn.GRU,
        GRUConfig(
            input_size=rnn_input_size,
            hidden_size=20,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
    )
elif rnn_type == 'rnn':
    rnn_cfg = FactoryConfig(
        torch.nn.RNN,
        ElmanConfig(
            input_size=rnn_input_size,
            hidden_size=20,
            num_layers=1,
            bias=True,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=False,
        )
    )
else:
    ValueError(f"Unrecognized value {rnn_type} for `rnn_type`.")

readout_network_cfg = FactoryConfig.from_class(
    FCN,
    FCNConfig(
        layer_sizes=[rnn_cfg.args_cfg.hidden_size, 80, 2],
        nonlinearities=[
            FactoryConfig.from_class(torch.nn.GELU, GELUConfig()), 
            FactoryConfig.from_class(torch.nn.Identity, IdentityConfig())
        ],
        dropouts=[0.5, None]
    )
)

model_cfg = AutoRNNConfig(
    input_network=input_network_cfg,
    rnn=rnn_cfg,
    readout_network=readout_network_cfg
)

# Convert, serialize/save, de-serialize, reconstruct.
cfg_filepath = output_dir / (filename + '.json')
reconstructed_model_cfg = serialize_and_deserialize(model_cfg, cfg_filepath)



# ---------------------------- Test instantiation --------------------------- #
# Replace 'embedding_dim' string place holder with mock embedding dim.
# Add an arbitrary integer for the embedding dimension in order to test model 
# instantiation.
mock_embedding_dim = 5
if input_network_cfg.args_cfg.layer_sizes is None:
    reconstructed_model_cfg.rnn.args_cfg.input_size = mock_embedding_dim
else:
    reconstructed_model_cfg.input_network.args_cfg.layer_sizes[0] = mock_embedding_dim

# Instantiate any FactoryConfig objects into objects of the associated class.
# reconstructed_model_cfg = r_utils.recursive(
#     reconstructed_model_cfg,
#     branch_conditionals=(
#         r_utils.dict_branch, 
#         r_utils.tuple_branch, 
#         r_utils.list_branch, 
#         r_utils.dataclass_branch_with_instantiation
#     ),
#     leaf_fns=(
#         lambda x: x,
#     )
# )
reconstructed_model_cfg = recursive_instantiation(reconstructed_model_cfg)

# Create mock tokens and set device.
mock_tokens = torch.randn((2, mock_embedding_dim))
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps:0' if torch.backends.mps.is_available() 
    else 'cpu'
)

# Test instantiation of model.
# model = build_model(reconstructed_model_cfg, mock_tokens)
model = AutoRNN(
    input_network=reconstructed_model_cfg.input_network,
    rnn=reconstructed_model_cfg.rnn,
    readout_network=reconstructed_model_cfg.readout_network,
    tokens=mock_tokens
)
print("Model successfully instantiated.")

# Try forward pass and generation on mock data.
mock_input = torch.randn((10, mock_embedding_dim))
forward_out = model(mock_input)
print(f"Output of forward pass on mock data: \n {forward_out}.")
generate_out = model.generate(mock_input)
print(f"Output of generation on mock data: \n {generate_out}.")