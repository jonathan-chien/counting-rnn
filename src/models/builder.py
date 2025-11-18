from .config import AutoRNNConfig, ModelConfig
from data import builder as data_builder
from .networks import AutoRNN
from general_utils import config as config_utils
from general_utils import ml as ml_utils


# def build_model(model_cfg: AutoRNNConfig, tokens, device) -> AutoRNN:
#     """ 
#     """
#     return AutoRNN(
#         input_network=model_cfg.input_network,
#         rnn=model_cfg.rnn,
#         readout_network=model_cfg.readout_network,
#         tokens=tokens
#     ).to(device)

def insert_embedding_dim(embedding_dim, auto_rnn_cfg):
    if auto_rnn_cfg.input_network.args_cfg.layer_sizes is None:
        auto_rnn_cfg.rnn.args_cfg.input_size = embedding_dim
    else:
        auto_rnn_cfg.input_network.args_cfg.layer_sizes[0] = embedding_dim
    return auto_rnn_cfg

def build_model_from_filepath(
    model_cfg_filepath, 
    data_cfg_filepath, 
    reproducibility_cfg_filepath,
    seed_idx, # This is passed to the logic for constructing sequences; if a random rotation of hypercube is used to get tokens, this will impact what count/EOS tokens are associated with a model
    device, 
    test_pass=False
):
    
    # Build sequences in order to get embedding dimension and tokens.
    sequences, data_cfg_dict, _ = data_builder.build_sequences_from_filepath(
        data_cfg_filepath=data_cfg_filepath, 
        reproducibility_cfg_filepath=reproducibility_cfg_filepath,
        build=['train'], # Any split will do, tokens should be the same across all splits
        seed_idx=seed_idx,
        print_to_console=False
    )
    embedding_dim = sequences['train'].transform.ambient_dim
    tokens = data_builder.get_autoregressive_tokens(sequences['train']).to(device)

    # Get 'recovery' seed to support reproducible model initialization (TODO:
    # encapsulate this pattern of deserializing a cfg with dicts).
    reproducibility_cfg_dict = {}
    reproducibility_cfg_dict['base'] = config_utils.serialization.deserialize(reproducibility_cfg_filepath)
    reproducibility_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(
        reproducibility_cfg_dict['base']
    )

    # Load in config.
    model_cfg_dict = {}
    model_cfg_dict['base'] = config_utils.serialization.deserialize(model_cfg_filepath)

    # Handle potential legacy cases where there is no initializtion_cfg and
    # model_cfg points to auto_rnn_cfg.
    if isinstance(model_cfg_dict['base'], AutoRNNConfig):
        model_cfg = ModelConfig(
            auto_rnn_cfg=model_cfg_dict['base'],
            initialization_cfg=ml_utils.config.InitializationConfig(steps=[])
        )
        model_cfg_dict['base'] = model_cfg

    # Insert embedding dimension. 
    model_cfg_dict['base'].auto_rnn_cfg = insert_embedding_dim(embedding_dim, model_cfg_dict['base'].auto_rnn_cfg)
    
    # Apply recovery seed immediately before recovering and instantiate model.
    ml_utils.reproducibility.apply_reproducibility_settings(
        reproducibility_cfg=reproducibility_cfg_dict['recovered'],
        seed_idx=seed_idx,
        split='recovery'
    )
    model_cfg_dict['recovered'] = config_utils.serialization.recursive_recover(model_cfg_dict['base'])
    model = AutoRNN(
        **config_utils.serialization.shallow_asdict(model_cfg_dict['recovered'].auto_rnn_cfg),
        tokens=tokens
    ).to(device)

    # If initilization_cfg.steps is empty list, function will not do anything.
    ml_utils.initialization.init_params_from_cfg(model, model_cfg_dict['recovered'].initialization_cfg)
    
    if test_pass:
        SEQ_IDX = 0
        input_ = sequences['train'][SEQ_IDX][0].to(device)
        print("Test forward pass and generation requested: \n")
        forward_out = model(input_)
        print(f"Output of forward pass: \n {forward_out}.")
        joined = model.generate(input_)
        print(f"Input joined with output of generation: \n {joined}.")

    return model, model_cfg_dict, data_cfg_dict, reproducibility_cfg_dict