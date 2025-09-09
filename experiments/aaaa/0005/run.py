from pathlib import Path

from datetime import datetime
from engine.driver import run
from general_utils import metadata as metadata_utils


# ----------------------------- Set run args -------------------------------- #
run_args = dict(
    data_train_cfg_ref_list=['aaaa/0000/0019'],
    model_cfg_ref_list=['aaaa/0001/0000'],
    pretrained_model_filepath_list=None,
    training_cfg_ref_list=['aaaa/0001/0001'],
    data_test_cfg_ref_list=['aaaa/0000/0019', 'aaaa/0000/0020'],
    testing_cfg_ref_list=['aaaa/0001/0000'],
    reproducibility_cfg_ref_list=['aa/0000'],
    seed_idx_list=[0, 1],
    exp_group_id='aaaa',
    exp_id='0005',
    run_id_suffix='',
    model_suffix='_best.pt',
    weights_only=False
)

# ------------------------ Collect and save metadata ------------------------ #
exp_dir = Path(__file__).resolve().parent
metadata_utils.collect_and_save_metadata(
    additional_info={'run_args': run_args, 'exp_dir': str(exp_dir)},
    filepath=exp_dir / 'metadata.json',
    enforce_clean_git_tree=False,
    overwrite=False
)
metadata_utils.create_textfile(
    """\
    Train 20 hidden unit (original model) network for max 500 epochs on train
    set consisting of demo and null tokens numbering from 1 to 19 odd, and test
    on demo and null tokens numbering from 0 to 20 even.
    """,
    filepath=exp_dir / 'README.md',
    dedent=True,
    overwrite=False,
)

# --------------------------------- Run ------------------------------------- #
training, testing, returned_exp_dir = run(**run_args)

# Verify that experimental results were logged in the same directory as this script.
if str(exp_dir) != ('/Users/jonathan/projects/counting_rnn/' + str(returned_exp_dir)):
    raise RuntimeError(
        "Mismatch between intended and actual location of experimental results.\n"
        f"- Expected: {exp_dir} (location of this script)\n"
        f"- Received: {returned_exp_dir} (returned by the run/run_curriculum function)\n\n"
    )



