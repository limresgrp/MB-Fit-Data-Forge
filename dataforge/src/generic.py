import json
import os
import re
import ast
import h5py
import numpy as np
from functools import reduce
from typing import Callable, List, Union


def append_suffix_to_filename(filename, suffix):
    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}{suffix}{extension}"
    return new_filename

def argofyinx(x, y):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    return np.ma.array(yindex, mask=mask)

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

def read_h5_file(h5_filepath: str):
    def load(s):
        d: dict = json.loads(s)
        for k, v in d.items():
            try:
                if isinstance(v, list):
                    d[k] = np.array(v)
            except:
                pass
        return d
    
    with h5py.File(h5_filepath, 'r') as h5f:
        all_coords       = h5f['coordinates'] [:]
        all_atom_types   = h5f['atom_types']  [:].astype("U")
        
        # Convert JSON strings back to list of dicts
        all_info_dicts_json = h5f['info_dicts'][:]
        if isinstance(all_info_dicts_json[0], bytes):
            all_info_dicts_json = [s.decode('utf-8') for s in all_info_dicts_json]
        all_info_dicts = [load(s) for s in all_info_dicts_json]
        
        extra_data = {}
        for k in h5f.keys():
            if k not in ['coordinates', 'atom_types', 'info_dicts']:
                try:
                    v = h5f[k][:]
                    if isinstance(v, np.ndarray):
                        v = v.astype("U")
                    extra_data[k] = v
                except:
                    extra_data[k] = h5f[k][()].decode('utf-8')

    return all_coords, all_atom_types, all_info_dicts, extra_data

def write_h5_file(
    h5_filepath: str,
    all_coords: np.ndarray,
    all_atom_types: np.ndarray,
    all_info_dicts: Union[dict, list[dict]],
    **kwargs
):
    with h5py.File(h5_filepath, 'w') as h5f:
        h5f.create_dataset('coordinates', data=all_coords)
        h5f.create_dataset('atom_types',  data=all_atom_types.astype(np.string_))
        
        # Convert list of dicts to JSON strings
        if isinstance(all_info_dicts, list):
            all_info_dicts_json = [json.dumps(d) for d in all_info_dicts]
            h5f.create_dataset('info_dicts', data=np.array(all_info_dicts_json, dtype=np.string_))
        else:
            h5f.create_dataset('info_dicts', data=json.dumps(all_info_dicts))
        
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                 h5f.create_dataset(k, data=v.astype(np.string_))
            else:
                h5f.create_dataset(k, data=json.dumps(v))