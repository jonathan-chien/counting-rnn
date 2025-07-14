import torch

from data.builder import build_split_sequences, get_autoregressive_tokens
from data.sequences import Sequences
from engine.config import DataLoaderConfig, EvalFnConfig, LoggerConfig, LossTermConfig, TestConfig
from engine.driver import run_testing_from_filepath
from engine.eval import evaluate
from engine.loss import LossTerm, spectral_entropy, wrapped_cross_entropy_loss
from engine.utils import Logger, compute_accuracy
from general_utils.config import CallableConfig, TorchDeviceConfig
from general_utils import fileio as fileio_utils
from general_utils import serialization as serialization_utils


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
        optimizer=None,
        mode='eval'
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
        optimizer=None,
        mode='eval'
    ),
    kind='class',
    recovery_mode='call'
)

logger = CallableConfig.from_callable(
    Logger,
    LoggerConfig(
        log_dir='',
        log_name='test',
        print_flush_epoch=False,
        print_flush_batch=False
    ),
    kind='class',
    recovery_mode='call'
)

dataloader = CallableConfig.from_callable(
    torch.utils.data.DataLoader,
    DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        collate_fn=CallableConfig.from_callable(
            Sequences.pad_collate_fn,
            args_cfg=None,
            kind='static_method',
            recovery_mode='get_callable'
        )
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

# Args for the eval.evaluate function called in the test.test function.
eval_fn_cfg = EvalFnConfig(
    dataloader=dataloader,
    switch_label='switch_label___',
    loss_terms=[loss_term_1],
    logger=logger,
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


test_cfg = TestConfig(
    eval_fn_cfg=eval_fn_cfg,
    test_split_seed_idx=0
)


base_dir = 'configs/testing'
sub_dir = '__00'
output_dir = fileio_utils.make_dir(base_dir, sub_dir)
filename = fileio_utils.make_filename('0000')

test_cfg_filepath = output_dir / (filename + '.json')
serialization_utils.serialize(test_cfg, test_cfg_filepath)



run_testing_from_filepath(
    data_cfg_filepath='configs/datasets/__00/0000.json',
    model_cfg_filepath='configs/models/__01/0000.json',
    # model_filepath='experiments/__00/0000/output/seed00/models/0_best.pt',
    test_cfg_filepath=test_cfg_filepath,
    run_dir='experiments/__00/0000/',
    model_suffix='_best.pt',
    seed_idx=0
)

# serialization_utils.deserialize(test_cfg_filepath)


# test_cfg = serialization_utils.recursive_recover(test_cfg)

# testing, test_cfg_dict, model_cfg_dict, data_cfg_dict











# # configs/datasets/000/000/000_000_000.json should point to dummy dataset.
# data_cfg = serialization_utils.deserialize('configs/datasets/0000/0000.json')
# data_cfg = serialization_utils.recursive_recover(data_cfg)

# sequences = build_split_sequences(
#     data_cfg,
#     build=['test'],
#     seed_ind=[0]
# )

# # Get embedding dimension and a sequence from sequences_cfg.
# embedding_dim = data_cfg.sequences_cfg.embedder.ambient_dim
# tokens = get_autoregressive_tokens(sequences['test']).to(test_cfg.eval_fn_cfg.device)
# seq, labels, _, _, _ = sequences['test'][0]
# seq = seq.to(test_cfg.eval_fn_cfg.device)

# # configs/models/000/000/000_000_000.json should point to a dummy model.
# model_cfg = serialization_utils.deserialize('configs/models/0000/0000.json')
# # model_cfg = serialization_utils.recursive_recover(model_cfg)
# model = test_model(
#     embedding_dim=embedding_dim, 
#     model_cfg=model_cfg,
#     tokens=tokens,
#     input_=seq,
#     device=test_cfg.eval_fn_cfg.device
# )

# # Add dataset to dataloaders.
# test_cfg.eval_fn_cfg.dataloader = test_cfg.eval_fn_cfg.dataloader.manually_recover(dataset = sequences['test'])

# # For testing, just run a few epochs.
# eval_fn_cfg_copy = copy.deepcopy(test_cfg.eval_fn_cfg)

# eval_fn_cfg_copy.num_epochs = 2

# # Replace placeholder.
# eval_fn_cfg_copy.switch_label = sequences['test'].special_tokens['switch']['label'].to(test_cfg.eval_fn_cfg.device)
# testing = evaluate(model, **serialization_utils.shallow_asdict(eval_fn_cfg_copy))