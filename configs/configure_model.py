from datetime import date

import torch

from models import builder as model_builder
from models.config import AutoRNNConfig, ElmanConfig, GRUConfig, FCNConfig, GELUConfig, IdentityConfig, ReLUConfig
from models.networks import FCN
from general_utils.config.types import CallableConfig
from general_utils import config as config_utils
from general_utils import fileio as fileio_utils


def main():
    # ---------------------------- Set directory ---------------------------- #
    base_dir = 'configs/models'
    sub_dir_1 = str(date.today())
    sub_dir_2 = 'b'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = fileio_utils.make_filename('0000')

    # ----------------------- Configure input network ----------------------- #
    input_network = CallableConfig.from_callable(
        FCN, 
        FCNConfig(
            # layer_sizes=['embedding_dim___', 10],
            layer_sizes=None,
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
    elif rnn_type == 'elman':
        rnn = CallableConfig.from_callable(
            torch.nn.RNN,
            ElmanConfig(
                input_size=rnn_input_size,
                hidden_size=100,
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
            # layer_sizes=[rnn.args_cfg.hidden_size, 20, 2],
            # nonlinearities=[
            #     CallableConfig.from_callable(torch.nn.GELU, GELUConfig(), kind='class', recovery_mode='call'), 
            #     CallableConfig.from_callable(torch.nn.Identity, IdentityConfig(), kind='class', recovery_mode='call')
            # ],
            # dropouts=[0.5, None]
            layer_sizes=[rnn.args_cfg.hidden_size, 2],
            nonlinearities=[
                CallableConfig.from_callable(torch.nn.Identity, IdentityConfig(), kind='class', recovery_mode='call'),
            ],
            dropouts=[None]
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
    model_cfg_filepath = output_dir / (filename + '.json')
    _ = config_utils.serialization.serialize(model_cfg, model_cfg_filepath)

    # ---------------------------- Test instantiation --------------------------- #
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() 
        else 'mps:0' if torch.backends.mps.is_available() 
        else 'cpu'
    )
    seed_idx = 0
    model_builder.build_model_from_filepath(
        model_cfg_filepath=model_cfg_filepath, 
        data_cfg_filepath='configs/datasets/0000-00-00/a/0000.json', 
        reproducibility_cfg_filepath='configs/reproducibility/0000-00-00/a/0000.json',
        seed_idx=seed_idx, 
        device=device,
        test_pass=True
    )

    # --------------------------- Summarize config --------------------------- #
    # Registry of items to extract from the config.
    REGISTRY = {
        'input_network': 'input_network.path',
        'input_network_layer_sizes': 'input_network.args_cfg.layer_sizes',
        'input_network_nonlinearities': 'input_network.args_cfg.nonlinearities',
        'input_network_dropouts' : 'input_network.args_cfg.dropouts',
        'rnn_type': 'rnn.path',
        'rnn_input_size': 'rnn.args_cfg.input_size',
        'rnn_hidden_size': 'rnn.args_cfg.hidden_size',
        'rnn_nonlinearity': (
            lambda model_cfg: config_utils.ops.traverse_dotted_path(model_cfg, 'rnn.args_cfg.nonlinearity')
            if config_utils.ops.traverse_dotted_path(model_cfg, 'rnn.path').endswith('RNN') 
            else None
        ),
        'readout_network': 'readout_network.path',
        'readout_network_layer_sizes': 'readout_network.args_cfg.layer_sizes',
        'readout_network_nonlinearities': 'readout_network.args_cfg.nonlinearities',
        'readout_network_dropouts': 'readout_network.args_cfg.dropouts',
    }


    # Summarize config to .xlsx file.
    config_utils.summary.summarize_cfg_to_xlsx(
        model_cfg_filepath,
        config_kind='models',
        config_id=str(model_cfg_filepath).removeprefix('configs/models/').removesuffix('.json'),
        dotted_path_registry=REGISTRY,
        note='',
        xlsx_filepath='configs/logs.xlsx'
    )

if __name__ == '__main__':
    main()
 

