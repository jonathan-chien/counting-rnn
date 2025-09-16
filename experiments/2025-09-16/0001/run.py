import os

from pathlib import Path

from engine.driver import run
from general_utils import metadata as metadata_utils


# ----------------------------- Set run args -------------------------------- #
run_args = dict(
    data_train_cfg_ref_list=['2025-09-05/o/0000'],
    model_cfg_ref_list=['2025-09-09/a/0002'],
    pretrained_model_filepath_list=None,
    training_cfg_ref_list=['2025-09-16/z/0000'],
    data_test_cfg_ref_list=['2025-09-05/o/0000', '2025-09-05/e/0000'],
    cross_test=True,
    testing_cfg_ref_list=['0000-00-00/a/0000'],
    reproducibility_cfg_ref_list=['0000-00-00/a/0000'],
    seed_idx_list=[0, 1, 2, 4, 5],
    exp_date='2025-09-16',
    exp_id='0001',
    run_id_suffix='',
    model_suffix='_best.pt',
    weights_only=False
)

# ------------------------ Collect and save metadata ------------------------ #
exp_dir = Path(__file__).resolve().parent
metadata_utils.collect_and_save_metadata(
    additional_info={'run_args': run_args, 'exp_dir': str(exp_dir)},
    filepath=exp_dir / 'metadata.json',
    enforce_clean_git_tree=True,
    overwrite=False
)
metadata_utils.create_textfile(
    """
    Repeat of experiment 2025-09-16/0000 but with only cross entropy loss.
    Very slow training was observed on 2025-09-16/0000. This experiment tests
    whether this might be due to the computation of spectral entropy for a 
    larger network.
    """,
    filepath=exp_dir / 'README.md',
    dedent=True,
    overwrite=False,
)

# --------------------------------- Run ------------------------------------- #
training, testing, returned_exp_dir = run(**run_args)

# Verify that experimental results were logged in the same directory as this script.
if str(exp_dir) != str(returned_exp_dir.resolve()):
    raise RuntimeError(
        "Mismatch between intended and actual location of experimental results.\n"
        f"- Expected: {exp_dir} (location of this script)\n"
        f"- Received: {returned_exp_dir} (returned by the run/run_curriculum function)\n\n"
    )
