import numpy as np

from . import validation as validation_utils


def recursive_seed_sequence_spawn(
    parent, 
    num_children_per_level, 
    dtype='uint32', 
    return_as='int',
    depth=0, # Internal use during recursion, don't pass in manually
):
    """ 
    Recursively creates arbitrarily deeply nested lists of seeds, where each
    level of the hierarchy is extended using the spawn method of the 
    SeedSequence class. Base case results in a seed returned (either as int or 
    numpy array), whereas all other calls return lists.
    """
    validation_utils.validate_iterable_contents(
        num_children_per_level,
        predicate=validation_utils.is_pos_int,
        expected_description="a positive int"
    )
    validation_utils.validate_nonneg_int(depth)
    
    if depth == len(num_children_per_level):
        # Base case.
        seed = parent.generate_state(n_words=1, dtype=dtype)
        if return_as == 'int':
            seed = int(seed[0])
        elif return_as != 'numpy':
            raise ValueError(
                "Invalid value for `return_as`: expected 'int' or 'numpy' "
                f"but got {return_as}."
            )
        return seed
    elif depth < len(num_children_per_level):
        # Recursion case.
        children = parent.spawn(num_children_per_level[depth])
        return [
            recursive_seed_sequence_spawn(
                parent=child, 
                num_children_per_level=num_children_per_level, 
                dtype=dtype, 
                return_as=return_as,
                depth=depth+1
            )
            for child in children
        ]
    else:
        raise ValueError(
            f"Invalid value for `depth`: got {depth} but expected at most "
            f"{len(num_children_per_level)}."
        )
    
def generate_seed_sequence(
    num_children_per_level, 
    entropy=None, 
    dtype='uint32', 
    return_as='int'
):
    """ 
    """
    root_seed_seq = np.random.SeedSequence(entropy=entropy) 

    seeds = recursive_seed_sequence_spawn(
        root_seed_seq,
        num_children_per_level,
        dtype=dtype,
        return_as=return_as
    )

    return seeds, root_seed_seq, root_seed_seq.entropy