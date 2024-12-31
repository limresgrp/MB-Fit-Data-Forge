import os
import re
import ast
import h5py
import numpy as np
from functools import reduce
from typing import Callable, List


def argofyinx(x, y):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    return np.ma.array(yindex, mask=mask)

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

def parse_string_to_dict(input_string: str) -> dict:
    """
    Parses a string representation of key-value pairs into a dictionary.

    The input string should have key-value pairs separated by commas, with keys and values
    separated by '='. Keys and values that are not lists or nested structures will be wrapped
    in quotes. Lists will be converted to NumPy arrays.

    Args:
        input_string (str): The input string containing key-value pairs.

    Returns:
        dict: A dictionary with keys and values parsed from the input string. Lists are converted
              to NumPy arrays.

    Raises:
        ValueError: If the input string cannot be parsed into a dictionary.

    Example:
        input_string = "key1=value1,key2=[1,2,3],key3=value3"
        result = parse_string_to_dict(input_string)
        # result will be {'key1': 'value1', 'key2': np.array([1, 2, 3]), 'key3': 'value3'}
    """
    # Preprocess the string to add quotes around unquoted keys and values
    def add_quotes(match):
        key, value = match.groups()
        # Wrap the key in quotes
        key = f"'{key}'"
        # Wrap the value in quotes if it's not a list or nested structure
        if not re.match(r"[\[{]", value):  # Skip lists or nested structures
            value = f"'{value}'"
        return f"{key}={value}"

    # Add quotes to keys and values
    preprocessed_string = re.sub(r"(\w+)=([^,]+)", add_quotes, input_string)

    # Replace '=' with ':' to mimic a dictionary structure
    dict_like_string = preprocessed_string.replace("=", ":")

    # Use ast.literal_eval to safely evaluate the string as a dictionary
    try:
        parsed_dict = ast.literal_eval(f"{{{dict_like_string}}}")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse the input string: {e}")

    # Process the dictionary to convert lists to NumPy arrays
    for key, value in parsed_dict.items():
        if isinstance(value, list):
            parsed_dict[key] = np.array(value)

    return parsed_dict

def parse_dict_to_string(input_dict: dict) -> str:
    string = ""
    for k, v in input_dict.items():
        string += f"{k}="
        if isinstance(v, str):
            string += f"{v}, "
        elif isinstance(v, np.ndarray):
            string += f"{v.tolist()}, "
    return string

def read_h5_file(h5_filepath: str):
    with h5py.File(h5_filepath, 'r') as h5f:
        all_coords       = h5f['coordinates'] [:]
        all_atom_types   = h5f['atom_types']  [:].astype("U")
        all_info_strings = h5f['info_strings'][:].astype("U")
    return all_coords, all_atom_types, all_info_strings

def write_h5_file(h5_filepath: str, all_coords: np.ndarray, all_atom_types: np.ndarray, all_info_strings: np.ndarray):
    with h5py.File(h5_filepath, 'w') as h5f:
        h5f.create_dataset('coordinates',  data=all_coords)
        h5f.create_dataset('atom_types',   data=all_atom_types.astype(np.string_))
        h5f.create_dataset('info_strings', data=all_info_strings.astype(np.string_))