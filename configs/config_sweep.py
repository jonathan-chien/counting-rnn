import argparse, subprocess
from datetime import date
import itertools
import shlex
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument('script', help="Path to configure_*.py")
    parser.add_argument('base_dir', help="Path to directory storing all configs of a certain kind (e.g. configs/models/)")
    parser.add_argument('--fname_start_idx', type=int, default=0, help="Starting filename index (e.g. 9).")
    parser.add_argument('--zfill', type=int, default=4)
    parser.add_argument('--sub_dir_1', default=str(date.today()))
    parser.add_argument('--sub_dir_2', default='a')

    # Fixed overrides.
    parser.add_argument('--pre', action='append', default=[], help="Extra fixed overrides (not swept) before config construction; this is repeatable.")
    parser.add_argument('--set', action='append', default=[], help="Extra fixed overrides (not swept) after config construction; this is repeatable.")
    
    # Swept overrides.
    parser.add_argument('--sweep_pre', nargs=4, action='append', default=[], metavar=("KEY", "START", "STOP", "STEP"), help="Sweep a config value over a numeric range before config construction; this is repeatable.")
    parser.add_argument('--sweep_set', nargs=4, action='append', default=[], metavar=("KEY", "START", "STOP", "STEP"), help="Sweep a config value over a numeric range after config construction; this is repeatable.")

    return parser.parse_args()

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

def main():
    args = parse_args()
    fname_start_idx = args.fname_start_idx

    sweeps = [] # Use list to explicitly preserve order, even though in later Python version dicts preserve insertion order
    for channel, collection in (('pre', args.sweep_pre), ('set', args.sweep_set)):
        for key, sta, sto, ste in collection:
            start, stop, step = int(sta), int(sto), int(ste)
            values = list(get_inclusive_range(start, stop, step))
            if not values:
                raise ValueError(f"Sweep {key} did not produce any values.")
            sweeps.append((channel, key, values))
    if not sweeps:
        sweeps = [('none', '__EMPTY__', [None])]

    values_list = [values for _, _, values in sweeps]
    for combo in itertools.product(*values_list):
        # Base command.
        command = [
            sys.executable, args.script, 
            "--base_dir", args.base_dir,
            "--sub_dir_1", args.sub_dir_1, 
            "--sub_dir_2", args.sub_dir_2,
            "--index", str(fname_start_idx), 
            "--zfill", str(args.zfill)
        ]

        # Fixed overrides.
        for a in args.pre:
            command += ["--pre", a]
        for a in args.set:
            command += ["--set", a]

        # Match each key (an item to be overriden) with one value from the 
        # current combination.
        for (chan, key, _), val in zip(sweeps, combo):
            if key == '__EMPTY__':
                continue
            if chan == 'pre':
                command += ["--pre", f"{key}={val}"]
            elif chan == 'set':
                command += ["--set", f"{key}={val}"]

        print(">>>", shlex.join(command))
        subprocess.run(command, check=True)
        
        fname_start_idx += 1

if __name__ == '__main__':
    main()


        

    

