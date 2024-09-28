import os
import numpy as np
from functools import reduce
from typing import Callable, List

def append_suffix_to_filename(filename, suffix):
    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}{suffix}{extension}"
    return new_filename

def dynamic_for_loop(iterable, num_for_loops, func: Callable, args: List = [], **extra_args):
        if num_for_loops == 1:
            for elem in iterable:
                new_args = args.copy()
                new_args.append(elem)
                extra_args = func(new_args, **extra_args)
            return extra_args
        for elem in iterable:
            new_args = args.copy()
            new_args.append(elem)
            extra_args = dynamic_for_loop(iterable[1:], num_for_loops-1, func, args=new_args, **extra_args)
        return extra_args

def union_rows_2d(arr1, arr2):
    nrows, ncols = arr1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [arr1.dtype]}

    C = np.union1d(arr1.view(dtype), arr2.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    return C.view(arr1.dtype).reshape(-1, ncols)

def intersect_rows_2d(arr1, arr2):
    nrows, ncols = arr1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [arr1.dtype]}

    C = np.intersect1d(arr1.view(dtype), arr2.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    return C.view(arr1.dtype).reshape(-1, ncols)

def parse_slice(slice_str: str):
    parts = slice_str.split(':')

    start = None if parts[0] == '' else int(parts[0])
    stop = None if parts[1] == '' else int(parts[1])
    step = None if len(parts) == 2 or parts[2] == '' else int(parts[2])

    return slice(start, stop, step)

def get_package_root(module_file_path: str):
    # Traverse upwards to the top-level package
    module_root_folder = os.path.dirname(module_file_path)
    while os.path.isfile(os.path.join(module_root_folder, '__init__.py')):
        module_root_folder = os.path.dirname(module_root_folder)

    return module_root_folder

FOLDER_REPLACEMENTS = [
    (os.path.join("trimers", "dimers"), "dimers"),
    (os.path.join("dimers", "monomers"), "monomers")
]

def apply_replacements_fp(input_string, replacements = FOLDER_REPLACEMENTS, n: int = 1):
    """
    Apply replacement operations using a functional programming approach.
    
    Parameters:
    - input_string (str): The original string.
    - replacements (list of tuples): A list of (old_word, new_word) tuples.
    - n (int): Number of times to apply the replacements.
    
    Returns:
    - str: The modified string after applying the replacements.
    """
    # Define a single pass of applying all replacements
    def apply_once(s):
        return reduce(lambda acc, pair: acc.replace(*pair), replacements, s)
    
    # Apply the pass n times using reduce over the string
    return reduce(lambda acc, _: apply_once(acc), range(n), input_string)