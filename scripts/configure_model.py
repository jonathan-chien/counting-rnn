from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

from models.networks import FCN, AutoRNN
from general_utils.config_types import ArgsConfig, ContainerConfig, CallableConfig
from general_utils.serialization import serialize, deserialize, recursive_instantiation
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


@dataclass
class FCNConfig(ArgsConfig):
    layer_sizes: Optional[List[int]]
    nonlinearities: List[Optional[CallableConfig]]
    dropouts: List[Optional[CallableConfig]]

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
    input_network: CallableConfig
    rnn: CallableConfig
    readout_network: CallableConfig


def build_model(cfg: AutoRNNConfig, tokens) -> AutoRNN:
    # input_network = cfg.input_network.call()
    # rnn = cfg.rnn.call()
    # readout_network = cfg.readout_network.call()
    # return AutoRNN(input_network, rnn, readout_network, tokens)
    components = {
        name : network.call() for name, network in asdict(cfg).items()
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

def insert_embedding_dim(embedding_dim, model_cfg):
    if model_cfg.input_network.args_cfg.layer_sizes is None:
        model_cfg.rnn.args_cfg.input_size = embedding_dim
    else:
        model_cfg.input_network.args_cfg.layer_sizes[0] = embedding_dim
    return model_cfg

def test_model(embedding_dim, model_cfg, tokens, input_, device):
    """ 
    """
    model_cfg = insert_embedding_dim(embedding_dim, model_cfg)
    model_cfg = recursive_instantiation(model_cfg)

    # Test instantiation of model.
    model = AutoRNN(
        input_network=model_cfg.input_network,
        rnn=model_cfg.rnn,
        readout_network=model_cfg.readout_network,
        tokens=tokens
    ).to(device)
    print("Model successfully instantiated.")

    # Try forward pass and generation on mock data.
    forward_out = model(input_)
    print(f"Output of forward pass on mock data: \n {forward_out}.")
    generate_out = model.generate(input_)
    print(f"Output of generation on mock data: \n {generate_out}.")


if __name__ == '__main__':
    # ------------------------------ Set directory ------------------------------ #
    output_dir, filename = io_utils.make_file_dir_and_id(
        base_dir='configs/models',
        sub_dir_1='000',
        sub_dir_2='000',
        file_ind = ('000', '000', '000')
    )

    # TODO: remove _cfg suffix or consider _factory, breaking convention only because this is a standalone variable (but probably better not to do this)
    input_network = CallableConfig.from_callable(
        FCN, 
        FCNConfig(
            layer_sizes=['embedding_dim', 50],
            nonlinearities=[CallableConfig.from_callable(torch.nn.ReLU, ReLUConfig(), kind='class')],
            dropouts=[None]
        ),
        kind='class'
    )

    rnn_input_size = (
        input_network.args_cfg.layer_sizes[-1] 
        if input_network.args_cfg.layer_sizes is not None 
        else 'embedding_dim'
    )

    rnn_type = 'gru'
    if rnn_type == 'gru':
        rnn = CallableConfig.from_callable(
            torch.nn.GRU,
            GRUConfig(
                input_size=rnn_input_size,
                hidden_size=20,
                num_layers=1,
                bias=True,
                batch_first=True,
                bidirectional=False,
            ),
            kind='class'
        )
    elif rnn_type == 'rnn':
        rnn = CallableConfig(
            torch.nn.RNN,
            ElmanConfig(
                input_size=rnn_input_size,
                hidden_size=20,
                num_layers=1,
                bias=True,
                nonlinearity='tanh',
                batch_first=True,
                bidirectional=False,
            ),
            kind='class'
        )
    else:
        ValueError(f"Unrecognized value {rnn_type} for `rnn_type`.")

    readout_network = CallableConfig.from_callable(
        FCN,
        FCNConfig(
            layer_sizes=[rnn.args_cfg.hidden_size, 80, 2],
            nonlinearities=[
                CallableConfig.from_callable(torch.nn.GELU, GELUConfig(), kind='class'), 
                CallableConfig.from_callable(torch.nn.Identity, IdentityConfig(), kind='class')
            ],
            dropouts=[0.5, None]
        ),
        kind='class'
    )

    model_cfg = AutoRNNConfig(
        input_network=input_network,
        rnn=rnn,
        readout_network=readout_network
    )

    # Convert, serialize/save, de-serialize, reconstruct.
    cfg_filepath = output_dir / (filename + '.json')
    _ = serialize(model_cfg, cfg_filepath)
    reconstructed_model_cfg = deserialize(cfg_filepath)



    # ---------------------------- Test instantiation --------------------------- #
    # Create mock tokens and set device.
    mock_embedding_dim = 5
    device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps:0' if torch.backends.mps.is_available() 
        else 'cpu'
    )
    mock_tokens = torch.randn((2, mock_embedding_dim)).to(device)
    mock_input = mock_input = torch.randn((10, mock_embedding_dim)).to(device)
    model = test_model(
        embedding_dim=mock_embedding_dim, 
        model_cfg=reconstructed_model_cfg,
        tokens=mock_tokens,
        input_=mock_input,
        device=device
        )
    # # Replace 'embedding_dim' string place holder with mock embedding dim.
    # # Add an arbitrary integer for the embedding dimension in order to test model 
    # # instantiation.
    # mock_embedding_dim = 5
    # if input_network.args_cfg.layer_sizes is None:
    #     reconstructed_model_cfg.rnn.args_cfg.input_size = mock_embedding_dim
    # else:
    #     reconstructed_model_cfg.input_network.args_cfg.layer_sizes[0] = mock_embedding_dim

    # # Instantiate any FactoryConfig objects into objects of the associated class.
    # # reconstructed_model_cfg = r_utils.recursive(
    # #     reconstructed_model_cfg,
    # #     branch_conditionals=(
    # #         r_utils.dict_branch, 
    # #         r_utils.tuple_branch, 
    # #         r_utils.list_branch, 
    # #         r_utils.dataclass_branch_with_instantiation
    # #     ),
    # #     leaf_fns=(
    # #         lambda x: x,
    # #     )
    # # )
    # reconstructed_model_cfg = recursive_instantiation(reconstructed_model_cfg)

    # # Create mock tokens and set device.
    # mock_tokens = torch.randn((2, mock_embedding_dim))
    # device = torch.device(
    #     'cuda' if torch.cuda.is_available() 
    #     else 'mps:0' if torch.backends.mps.is_available() 
    #     else 'cpu'
    # )

    # # Test instantiation of model.
    # # model = build_model(reconstructed_model_cfg, mock_tokens)
    # model = AutoRNN(
    #     input_network=reconstructed_model_cfg.input_network,
    #     rnn=reconstructed_model_cfg.rnn,
    #     readout_network=reconstructed_model_cfg.readout_network,
    #     tokens=mock_tokens
    # )
    # print("Model successfully instantiated.")

    # # Try forward pass and generation on mock data.
    # mock_input = torch.randn((10, mock_embedding_dim))
    # forward_out = model(mock_input)
    # print(f"Output of forward pass on mock data: \n {forward_out}.")
    # generate_out = model.generate(mock_input)
    # print(f"Output of generation on mock data: \n {generate_out}.")