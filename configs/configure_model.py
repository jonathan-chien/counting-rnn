import torch

from models import builder as model_builder
from models.config import AutoRNNConfig, ElmanConfig, GRUConfig, FCNConfig, GELUConfig, IdentityConfig, ReLUConfig
from models.networks import FCN
from general_utils.config import CallableConfig
from general_utils import serialization as serialization_utils
from general_utils import fileio as fileio_utils


if __name__ == '__main__':
    # ---------------------------- Set directory ---------------------------- #
    base_dir = 'configs/models'
    sub_dir_1 = 'aaaa'
    sub_dir_2 = '0001'
    output_dir = fileio_utils.make_dir(base_dir, sub_dir_1, sub_dir_2)
    filename = fileio_utils.make_filename('0000')

    # ----------------------- Configure input network ----------------------- #
    input_network = CallableConfig.from_callable(
        FCN, 
        FCNConfig(
            layer_sizes=['embedding_dim___', 10],
            # layer_sizes=None,
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
                hidden_size=1,
                num_layers=1,
                bias=True,
                batch_first=True,
                bidirectional=False,
            ),
            kind='class',
            recovery_mode='call'
        )
    elif rnn_type == 'rnn':
        rnn = CallableConfig.from_callable(
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
            layer_sizes=[rnn.args_cfg.hidden_size, 20, 2],
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
    model_cfg_filepath = output_dir / (filename + '.json')
    _ = serialization_utils.serialize(model_cfg, model_cfg_filepath)

    # ---------------------------- Test instantiation --------------------------- #
    device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps:0' if torch.backends.mps.is_available() 
        else 'cpu'
    )
    seed_idx = 0
    model_builder.build_model_from_filepath(
        model_cfg_filepath=model_cfg_filepath, 
        data_cfg_filepath='configs/datasets/demo/0000/0000.json', 
        reproducibility_cfg_filepath='configs/reproducibility/aaaa/0000/0000.json',
        seed_idx=seed_idx, 
        device=device,
        test_pass=True
    )
 