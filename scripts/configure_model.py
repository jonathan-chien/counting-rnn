from typing import List, Optional

import torch

from models import builder as model_builder
from models.config import AutoRNNConfig, ElmanConfig, GRUConfig, FCNConfig, GELUConfig, IdentityConfig, ReLUConfig
from models.networks import FCN
from general_utils.config import CallableConfig
from general_utils.serialization import serialize, deserialize, recursive_recover
from general_utils import fileio as io_utils


if __name__ == '__main__':
    # ---------------------------- Set directory ---------------------------- #
    output_dir, filename = io_utils.make_file_dir_and_id(
        base_dir='configs/models',
        sub_dir_1='000',
        sub_dir_2='000',
        file_ind = ('000', '000', '000')
    )

    # ----------------------- Configure input network ----------------------- #
    input_network = CallableConfig.from_callable(
        FCN, 
        FCNConfig(
            layer_sizes=['embedding_dim___', 50],
            nonlinearities=[CallableConfig.from_callable(torch.nn.ReLU, ReLUConfig(), kind='class', recovery_mode='call')],
            dropouts=[None]
        ),
        kind='class',
        recovery_mode='call'
    )

    # ---------------------------- Configure RNN ---------------------------- #
    rnn_input_size = (
        input_network.args_cfg.layer_sizes[-1] 
        if input_network.args_cfg.layer_sizes is not None 
        else 'embedding_dim___'
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
            kind='class',
            recovery_mode='call'
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
            kind='class',
            recovery_mode='call'
        )
    else:
        ValueError(f"Unrecognized value {rnn_type} for `rnn_type`.")

    # ---------------------- Configure readout network ---------------------- #
    readout_network = CallableConfig.from_callable(
        FCN,
        FCNConfig(
            layer_sizes=[rnn.args_cfg.hidden_size, 80, 2],
            nonlinearities=[
                CallableConfig.from_callable(torch.nn.GELU, GELUConfig(), kind='class', recovery_mode='call'), 
                CallableConfig.from_callable(torch.nn.Identity, IdentityConfig(), kind='class', recovery_mode='call')
            ],
            dropouts=[0.5, None]
        ),
        kind='class',
        recovery_mode='call'
    )

    # ------------------------- Instantiate and save ------------------------ #
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
    model = model_builder.test_model(
        embedding_dim=mock_embedding_dim, 
        model_cfg=reconstructed_model_cfg,
        tokens=mock_tokens,
        input_=mock_input,
        device=device
        )
 