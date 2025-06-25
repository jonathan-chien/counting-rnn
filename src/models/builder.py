from dataclasses import asdict

from .config import AutoRNNConfig
from .networks import AutoRNN
from general_utils import serialization


def build_model(cfg: AutoRNNConfig, tokens) -> AutoRNN:
    """ 
    """
    components = {
        name : network.call() for name, network in asdict(cfg).items()
    }
    return AutoRNN(**components, tokens=tokens)

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
    model_cfg = serialization.recursive_instantiation(model_cfg)

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