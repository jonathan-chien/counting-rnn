import torch

from data import builder as data_builder 
from data.config import DataConfig, HypercubeConfig, EmbedderConfig, SequencesConfig, SeqLengths, NormalDistrConfig
from data.sequences import Hypercube, Embedder
from data import utils as data_utils
from general_utils.config import ReproducibilityConfig, SeedConfig, TorchDeterminismConfig, CallableConfig, TensorConfig
from general_utils import fileio as fileio_utils
from general_utils import reproducibility as reproducibility_utils
from general_utils import serialization as serialization_utils


# ------------------------------ Set directory ------------------------------ #
# output_dir, filename = fileio.make_file_dir_and_id(
#     base_dir='configs/datasets',
#     sub_dir_1='000',
#     sub_dir_2='000',
#     file_ind = ('000', '000', '000')
# )
base_dir = 'configs/datasets'
sub_dir = '000'
output_dir = fileio_utils.make_dir(base_dir, sub_dir)
filename = fileio_utils.make_filename('000')

# ------------------------ Reproducibility settings ------------------------- #
# 3 children for train, val, and test. num_words_per_child = number of desired
# seed repeats.
NUM_SEEDS = 50
seed_lists, entropy, _, _ = reproducibility_utils.generate_numpy_seed_sequence(
    num_children=3, num_words_per_child=NUM_SEEDS, dtype='uint32', return_as='int'
)

reproducibility_cfg = ReproducibilityConfig(
    entropy=entropy,
    seed_cfg_dict={
        split : [
            SeedConfig(torch_seed=s, cuda_seed=s) 
            for s in seeds
        ]
        for (split, seeds) in zip(('train', 'val', 'test'), seed_lists)
    },
    torch_determinism_cfg_dict={
        'train' : TorchDeterminismConfig(
            use_deterministic_algos=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True
        ),
        'val' : TorchDeterminismConfig(
            use_deterministic_algos=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True
        ),
        'test' : TorchDeterminismConfig(
            use_deterministic_algos=False,
            cudnn_deterministic=False,
            cudnn_benchmark=True
        )
    }
)

# ------------------------- Build auxiliary objects ------------------------- #
hypercube_args_cfg = HypercubeConfig(
    num_dims=2,
    coords=TensorConfig.from_tensor(
        torch.tensor([0, 1], dtype=torch.int64)
    ),
    inclusion_set=TensorConfig.from_tensor(
        torch.tensor(
            [[1, 0], [1, 1]],
            dtype=torch.int8
        )
    ),
    encoding=TensorConfig.from_tensor(
        torch.tensor([0, 1], dtype=torch.int8)
    )
)

seq_lengths = SeqLengths(
    lengths={
        'pos' : {
            'support' : TensorConfig.from_tensor(torch.arange(0, 10)),
            'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(10))
        },
        'neg' : {
            'support' : TensorConfig.from_tensor(torch.arange(0, 5)),
            'pmf' : TensorConfig.from_tensor(data_utils.uniform_pmf(5))
        }
    }
)

embedder_cfg = EmbedderConfig(
    ambient_dim=5,
    mean_center=False,
    offset_1=TensorConfig.from_tensor(-torch.tile(torch.tensor([0.5]), (hypercube_args_cfg.num_dims + 3,))), # Plus 3 for the dimensions corresponding to special tokens
    offset_2=None,
    method='random_rotation',
    noise_distr=CallableConfig.from_callable(
        torch.distributions.Normal, 
        NormalDistrConfig(loc=0, scale=0.05),
        kind='class',
        recovery_mode='call'
    )
)

# ----------------------------- Sequences config ---------------------------- #
sequences_cfg = SequencesConfig(
    num_seq='num_seq___',
    seq_order='permute',
    seq_lengths=seq_lengths,
    elem=CallableConfig.from_callable(
        Hypercube, 
        hypercube_args_cfg, 
        kind='class', 
        recovery_mode='call'
    ),
    embedder=CallableConfig.from_callable(
        Embedder,
        embedder_cfg,
        kind='class',
        recovery_mode='call'
    )
)

# ------------------------------- Serialize --------------------------------- #
# Attempt to serialize and reconstruct full cfg tree, and use reconstructed
# version to build, to ensure future reproducibility.
data_cfg = DataConfig(
    sequences_cfg=sequences_cfg,
    reproducibility_cfg=reproducibility_cfg
)

cfg_filepath = output_dir / (filename + '.json')
_ = serialization_utils.serialize(data_cfg, cfg_filepath)







reconstructed_data_cfg = serialization_utils.deserialize(cfg_filepath)

# Recursive instantiation.
reconstructed_data_cfg = serialization_utils.recursive_recover(reconstructed_data_cfg)

# ----------------------------- Build sequences ----------------------------- #
# See if sequences build without error. Arbitrarily using the train split's 
# first seed and determinism settings here.
reproducibility_utils.set_seed(
    **serialization_utils.shallow_asdict(reconstructed_data_cfg.reproducibility_cfg.seed_cfg_dict['train'][0])
)
reproducibility_utils.set_torch_determinism(
    **serialization_utils.shallow_asdict(reconstructed_data_cfg.reproducibility_cfg.torch_determinism_cfg_dict['train'])
)

# Choose arbitrary split size.
reconstructed_data_cfg.sequences_cfg.num_seq = 1000

sequences = data_builder.build_hypercube_sequences(reconstructed_data_cfg.sequences_cfg) 

PRINT_TO_CONSOLE = True
if PRINT_TO_CONSOLE:
    # Retrieve sample from current split.
    seq_idx = 7
    seq, labels, _, _, _ = sequences[seq_idx]

    # Get version where all labels in positive vs negative class are the same.
    labels_uniform_class = labels.clone()
    labels_uniform_class[labels >= 4] = 4
    labels_uniform_class[labels < 0] = -1

    print(f"Shape of single sequence: {seq.shape}.")
    print(f"Shape of corresponding labels: {labels.shape}.")
    print(f"Sequence: \n {seq}.") 
    print(f"Labels by class: \n {labels_uniform_class}.") 
    print(f"Labels by stimulus: \n {labels}.") 
    print("\n")

# Optionally save sequences.
SAVE_DATASET = False
if SAVE_DATASET:
    dataset_filepath = output_dir / (filename + '.pt')
    fileio_utils.torch_save(sequences, dataset_filepath)






