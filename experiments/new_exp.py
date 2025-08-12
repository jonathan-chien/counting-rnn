import argparse, re
from datetime import date
from pathlib import Path
import os
import stat
from string import Template
import textwrap
from typing import Optional, Union
import warnings

from general_utils import validation as validation_utils


RUN_PY_TEMPLATES = {
    'standard': Template(textwrap.dedent(
        '''\
        import os

        from pathlib import Path

        from engine.driver import run
        from general_utils import metadata as metadata_utils


        # ----------------------------- Set run args -------------------------------- #
        run_args = dict(
            data_train_cfg_ref_list=[],
            model_cfg_ref_list=[],
            pretrained_model_filepath_list=None,
            training_cfg_ref_list=[],
            data_test_cfg_ref_list=[],
            testing_cfg_ref_list=[],
            reproducibility_cfg_ref_list=[],
            seed_idx_list=$seed_idx_list,
            exp_date=$exp_date,
            exp_id=$exp_id,
            run_id_suffix=$run_id_suffix,
            model_suffix=$model_suffix,
            weights_only=$weights_only
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
            """$readme_contents""",
            filepath=exp_dir / 'README.md',
            dedent=True,
            overwrite=False,
        )

        # --------------------------------- Run ------------------------------------- #
        training, testing, returned_exp_dir = run(**run_args)

        # Verify that experimental results were logged in the same directory as this script.
        if str(exp_dir) != str(returned_exp_dir):
            raise RuntimeError(
                "Mismatch between intended and actual location of experimental results.\n"
                f"- Expected: {exp_dir} (location of this script)\n"
                f"- Received: {returned_exp_dir} (returned by the run/run_curriculum function)\n\n"
            )
        '''
    ))
}

def make_file_executable(filepath: Union[Path, str]):
    """Make a file executable."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    
    # Make the file executable
    os.chmod(filepath, os.stat(filepath).st_mode | stat.S_IXUSR)
    return filepath

def is_four_digit_str(x: str) -> bool:
    """Check if a string is a four-digit number."""
    return bool(re.fullmatch(r'\d{4}', x))

def nd_str(x: int, n: int) -> str:
    """Convert an integer to a string of length n, padding with zeros if necessary."""
    return str(x).zfill(n)

def get_exp_ids(date_dir: Path, is_valid_dir=is_four_digit_str):
    """ 
    Returns list of elements of A = {x | x in {0, 1, ..., 9}^4 and x in
    date_dir}. Note that max(A) + 1 is the smallest "safe" 4 digit index such
    that an arbitrary number of new experiment IDs can be created with no
    collision risk.

    Parameters:
    -----------          
        date_dir : Path
            Directory containing experiment subdirectories named with 4-digit IDs.
        is_valid_dir : (callable)
            Function to validate directory names. No default lambda function
            value, as this can break pickling and de/serialization via
            CallableConfigs. Defaults to checking if the name is a 4-digit string.

    Returns:
    --------
        ids : list of int
            List of existing experiment IDs as integers.
        subdirs : list of pathlib.Path
            List of existing experiment directories as Path objects.
    """
    if not date_dir.exists():
        raise FileNotFoundError(f"Directory {date_dir} does not exist.")
    
    subdirs = [d for d in date_dir.iterdir() if d.is_dir() and is_valid_dir(d.name)]
    ids = [int(d.name) for d in subdirs]
    return ids, subdirs

def make_experiments(
    base_dir: Union[Path, str], 
    date_str: str, 
    exp_id_start: Optional[int] = None, 
    num_exps: int = 1, 
    seed_idx_list: Optional[list] = None,
    run_id_suffix: str = '',
    model_suffix: str = '_best.pt',
    weights_only: bool = False,
    readme_contents: str = ''
):
    """ 
    Create a series of experiment directories with a run.py file in each.
    The run.py file is populated with a template string, which can be customized
    using the appropriate kwargs.
    
    Parameters:
    -----------
        base_dir : Path, str
            Base directory for experiments.
        date_str : str
            Date string in the format YYYY-MM-DD.
        exp_id_start : int, optional
            Starting experiment ID. Defaults to None in which case k+1 is used, 
            where k is the maximum existing experiment ID in the date directory.
        num_exps : int
            Number of experiments to create (incrementing from exp_id_start by 1). Defaults to 1.
        seed_idx_list : list, optional
            List of seed indices to use in the run.py file. Defaults to None, 
            which will cause seed_idx_list to be set to [0].
        run_id_suffix : str
            Suffix to append to the run ID in the run.py file. Defaults to ''.
        model_suffix : str
            Suffix for the model file in the run.py file. Defaults to '_best.pt'.
        weights_only : bool
            Whether to save only the weights of the model. Defaults to False.

    Returns:
    --------
        None
    """
    base_dir = Path(base_dir)
    date_dir =  Path(base_dir) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    # Validation.
    validation_utils.validate_pos_int(num_exps)
    existing_ids, _ = get_exp_ids(date_dir)
    if exp_id_start is None:
        exp_id_start = max(existing_ids) + 1 if existing_ids else 0
    else:
        validation_utils.validate_nonneg_int(exp_id_start)
        requested_ids = {exp_id_start + k for k in range(num_exps)}
        conflicts = requested_ids & set(existing_ids)
        if conflicts:
            raise ValueError(
                f"Experiment IDs were requested in {date_dir} in the range {nd_str(exp_id_start, 4)} "
                f"to {nd_str(exp_id_start + num_exps - 1, 4)}, but the following IDs already exist: "
                f"{', '.join(map(lambda x: nd_str(x, 4), sorted(conflicts)))}. Please choose a different "
                "starting ID or number of experiments."
            )
    if seed_idx_list is None:
        seed_idx_list = [0]
    else:
        validation_utils.validate_iterable_contents(
            seed_idx_list, 
            predicate=validation_utils.is_nonneg_int,
            expected_description="a non-negative int"
        )
    
    # Create experiment directories and run.py files.
    for i_exp in range(num_exps):
        exp_id = nd_str(exp_id_start + i_exp, n=4)
        exp_dir = date_dir / exp_id
        exp_dir.mkdir(parents=False, exist_ok=False)

        run_py = exp_dir / 'run.py'
        with open(run_py, 'w') as f:
            f.write(RUN_PY_TEMPLATES['standard'].substitute(
                exp_date=repr(date_str), 
                exp_id=repr(exp_id), 
                seed_idx_list=repr(seed_idx_list),
                run_id_suffix=repr(run_id_suffix),
                model_suffix=repr(model_suffix),
                weights_only=str(bool(weights_only)),
                readme_contents=readme_contents.replace('$', '$$')
            ))
        make_file_executable(run_py)

        print(f"Created experiment directory: {exp_dir} with run.py")
        
def main():
    """Main function to parse arguments and create experiments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', default='experiments', type=str, help="Base directory for experiments."
    )
    parser.add_argument(
        '--date_str', default=str(date.today()), type=str, help="Date string in the format YYYY-MM-DD."
    )
    parser.add_argument(
        '--exp_id_start', default=None, type=int, help="Starting experiment ID. Defaults to None."
    )
    parser.add_argument(
        '--num_exps', default=1, type=int, help="Number of experiments to create. Defaults to 1."
    )
    parser.add_argument(
        '--seed_idx_list', default=[0], nargs='+', type=int, 
        help="List of seed indices to use in the run.py file. Defaults to [0]."
    )
    parser.add_argument(
        '--run_id_suffix', default='', type=str, help="Suffix to append to the run ID in the run.py file."      
    )
    parser.add_argument(
        '--model_suffix', default='_best.pt', type=str, 
        help="Suffix for the model file in the run.py file. Defaults to '_best.pt'."
    )
    parser.add_argument(
        '--weights_only', action='store_true', 
        help="Whether to save only the weights of the model. Defaults to False."
    )
    parser.add_argument(
        '--readme_contents', default='', type=str, 
        help=(
            "Contents to write in the README.md file in each experiment directory, "
            "or '@path/to/file.md' to load from a file. If a literal '@' is desired "
            "at the start of the README, write '@@' to escape."
        )
    )
    args = parser.parse_args()

    # Handle optional loading of README contents from a file.
    if args.readme_contents.startswith('@@'):
        # Escape the '@' character.
        args.readme_contents = args.readme_contents[1:] 
    if args.readme_contents.startswith('@'):
        # Load contents from provided file path.
        readme_filepath = Path(args.readme_contents[1:])
        if not readme_filepath.exists():
            raise FileNotFoundError(f"--readme_contents file {readme_filepath} does not exist.")
        args.readme_contents = readme_filepath.read_text(encoding='utf-8')

    make_experiments(**vars(args))


if __name__ == '__main__':
    main()
    
