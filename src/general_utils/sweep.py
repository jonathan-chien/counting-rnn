import argparse 
from dataclasses import dataclass, fields
from datetime import date
import itertools
import shlex
import subprocess
import sys 
from typing import List, Sequence, Tuple, Union


@dataclass
class ParamSweep: 
    channel: str
    key: str 
    values: Sequence[Union[int, float, str]]


@dataclass
class GridSweep: 
    script: str 
    base_dir: str 
    sub_dirs: Sequence[str]
    start_idx: int = 0
    zfill: int = 4
    fixed: Sequence[Tuple[str, str]] = ()
    sweeps: Sequence[ParamSweep] = ()
    python_exe: str = sys.executable
    echo_cmd: bool = True
    check_exit_code: bool = True
    dry_run: bool = False

    def __post_init__(self):
        """
        Normalize to immutable containers.
        """
        # Sequence of lists accepted for `fixed` but normalized to tuple of tuples.
        self.fixed = tuple(tuple(pair) for pair in self.fixed)
        self.sweeps = tuple(self.sweeps)


def get_combo_cmd(grid_sweep: GridSweep):
    """ 
    """
    # Get list of Sweep objects.
    values_lists = [sweep.values for sweep in grid_sweep.sweeps]
    combinations = itertools.product(*values_lists) if values_lists else [()]
    
    # Get starting index, increment by one for each combo.
    idx = grid_sweep.start_idx

    for combo in combinations:
        cmd: List[str] = [
            grid_sweep.python_exe,
            grid_sweep.script, 
            grid_sweep.base_dir, 
            '--sub_dirs', *grid_sweep.sub_dirs,
            '--idx', str(idx),
            '--zfill', str(grid_sweep.zfill),
        ]

        for channel, arg in grid_sweep.fixed:
            cmd += [f'--{channel}', arg]
        
        # Match each sweep object (corresponding to one param) with one item 
        # in product; this is skipped for empty grid_sweep.sweeps.
        for sweep, value in zip(grid_sweep.sweeps, combo):
            cmd += [f'--{sweep.channel}', f'{sweep.key}={value}']
            
        yield cmd, idx
            
        idx += 1

def run_grid_sweep(grid_sweep: GridSweep, passthrough: list[str] | None = None):
    """ 
    """
    passthrough = passthrough or []

    # Get Cartesian product of all parameter sweep values and run script for each combo.
    results = []
    for cmd, _ in get_combo_cmd(grid_sweep):
        cmd_with_passthrough = cmd + passthrough
        if grid_sweep.echo_cmd:
            print('>>>', shlex.join(cmd_with_passthrough))
        if grid_sweep.dry_run:
            continue

        out = subprocess.run(cmd_with_passthrough, check=grid_sweep.check_exit_code)
        results.append(out)

    return results
    

# ---------------------------- CLI helpers ---------------------------------- #
def build_parent_parser():
    return argparse.ArgumentParser(add_help=False)
    
def add_sweep_args(parser: argparse.ArgumentParser):
    """ 
    """
    # Execution/directory args.
    parser.add_argument('script', type=str, help="Path to script to be executed.")
    parser.add_argument('base_dir', type=str, help="Base directory for building path to where target script will write results.")
    parser.add_argument('--sub_dirs', nargs='+', type=str, default=[str(date.today()), 'a'], help="Optional path extensions added to <base_dir>/ to build directory where target script will write results.")

    # Fixed override args.
    parser.add_argument('--fixed', nargs=2, action='append', default=[], metavar=('CHANNEL', 'ARG'))

    # Swept overrides.
    parser.add_argument('--sweep', nargs=5, action='append', default=[], metavar=('CHANNEL', 'KEY', 'START', 'STOP', 'STEP'))
    parser.add_argument('--start_idx', type=int, default=0, help="Start value for index over all combinations in cartesian product.")
    parser.add_argument('--zfill', type=int, default=4, help="Int padding value for converting indices into n-digit strings, e.g. for building file/dir names.")

    # Runtime args.
    parser.add_argument('--no_echo_cmd', dest='echo_cmd', action='store_false', default=True, help="Suppress printing of command run by subprocess.run() to the console.")
    parser.add_argument('--no_exit_code_check', dest='check_exit_code', default=True, action='store_false', help="Specify not to raise on a non-zero exit code from subprocess.run().")
    parser.add_argument('--dry_run', action='store_true', help="Run `run_grid_sweep_from_parsed_args` but without actually calling subprocess.run().")

    return parser

def get_inclusive_range(start: int, stop: int, step: int):
    # Validate step.
    if step == 0:
        raise ValueError("Step cannot be 0.")
    if (stop - start) * step < 0:
        raise ValueError(
            f"`step` ({step}) has the wrong sign for range `start` ({start}) to `stop` ({stop})."
        )
    
    # Include stop index itself in range.
    end = stop + (1 if step > 0 else -1)

    return range(start, end, step)

def make_int_sweep(channel: str, key: str, start: int, stop: int, step: int):
    """ 
    """
    values = list(get_inclusive_range(start=start, stop=stop, step=step))
    if not values:
        raise ValueError(
            f"No values produced for channel '{channel}' and key '{key}'; "
            f"got start={start}, stop={stop}, step={step}."
        )
    return ParamSweep(
        channel=channel,
        key=key,
        values=values
    )

def parse_numeric_sweeps(args) -> Sequence[ParamSweep]:
    """ 
    """
    sweeps_list = []
    for channel, key, start, stop, step in args.sweep:
        sweeps_list.append(
            make_int_sweep(
                channel=channel,
                key=key,
                start=int(start),
                stop=int(stop),
                step=int(step)
            )
        )
    return sweeps_list

def run_grid_sweep_from_parsed_args(args, passthrough: list[str] | None = None):
    """ 
    """
    field_names = [f.name for f in fields(GridSweep)]
    args_with_parsed_sweeps = {
        key: val for key, val in vars(args).items() if key in field_names
    }
    args_with_parsed_sweeps['sweeps'] = parse_numeric_sweeps(args)

    return run_grid_sweep(GridSweep(**args_with_parsed_sweeps), passthrough)


if __name__ == '__main__':
    parser = build_parent_parser()
    parser = add_sweep_args(parser)
    args, passthrough = parser.parse_known_args()
    run_grid_sweep_from_parsed_args(args, passthrough)
