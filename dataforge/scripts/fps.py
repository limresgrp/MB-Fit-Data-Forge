import functools
import os
import glob
import argparse
from typing import Callable
import numpy as np
from itertools import permutations, product
from sklearn_extra.cluster import KMedoids
from multiprocessing import Pool
from dataforge.src import DataDict
from dataforge.src.generic import read_h5_file

def generate_permutations(names):
    unique_names = np.unique(names)
    perm_dict = {}
    for name in unique_names:
        indices = np.where(names == name)[0]
        perm_dict[name] = list(permutations(indices))
    return perm_dict

def apply_permutation(coords, perm):
    if len(perm) == 1:
        return coords
    permuted_coords = coords.copy()
    indices = np.array(perm)
    permuted_coords[:, np.sort(indices)] = coords[:, indices]
    return permuted_coords

def compute_frame_permutations(coords, symmetric_permutations):
    print("Computing permuted coordinates")
    all_permuted_coords = []
    for perm in symmetric_permutations:
        permuted_coords = coords.copy()
        for p in perm:
            permuted_coords = apply_permutation(permuted_coords, p)
        all_permuted_coords.append(permuted_coords)
    return np.stack(all_permuted_coords, axis=1)

def compute_pairwise_distances(A):
    print("Computing atoms pairwise distances")
    _, _, atoms, _ = A.shape
    # Compute the pairwise differences
    diff = A[:, :, :, np.newaxis, :] - A[:, :, np.newaxis, :, :]
    # Compute the squared distances
    dist_squared = np.sum(diff**2, axis=-1)
    # Extract the lower triangular part of the distance matrix (excluding the diagonal)
    i_lower = np.tril_indices(atoms, -1)
    descriptors_squared = dist_squared[:, :, i_lower[0], i_lower[1]]
    # Take the square root to get the Euclidean distances
    return np.sqrt(descriptors_squared)

def compute_distance_between_frames(descriptors1, descriptors2):
    diff = descriptors1[:, np.newaxis, :] - descriptors2[:, np.newaxis, :, :]
    # Compute the squared distances
    dist_squared = np.sum(diff**2, axis=-1)
    return dist_squared.min(axis=2).min(axis=1)

def compute_permutation_invariant_distance(point_idx, descriptors):
    return compute_distance_between_frames(descriptors[point_idx, :1], descriptors)

def fps(num_points: int, num_samples: int, compute_distance: Callable):
    assert num_samples < num_points, "The number of samples must be less than the number of points"
    
    # Initialize the array of minimum distances to the selected points
    min_distances = np.ones(num_points, dtype=np.float16) * np.inf
    # Initialize the list of selected points with the medoids
    selected_points = []

    def add_point(point):
        selected_points.append(point)
        min_distances[point] = 0.
    
    def update_min_distances(min_distances, farthest_point):
        distances_from_point = compute_distance(farthest_point)
        return np.minimum(min_distances, distances_from_point)

    farthest_point = np.random.randint(0, num_points)
    add_point(farthest_point)
    
    print("Start FPS")
    for i in range(len(selected_points), num_samples):
        # Compute distance of all points from newly sampled point and update min_distances array
        min_distances = update_min_distances(min_distances, farthest_point)

        # Find the point that is farthest from the selected points
        farthest_point = np.argmax(min_distances)
        add_point(farthest_point)
        if (i+1)%(num_samples//10) == 0:
            print(f"{i+1}/{num_samples} points sampled")
    print(f"{num_samples} points sampled!")
    
    return np.array(selected_points)

def furthest_point_sampling(num_samples, coords, names, chunk_max_dim=10000, max_symm_perm=100):
    points = np.arange(len(coords))
    print("Generating permutations")
    perm_dict = generate_permutations(names)
    symmetric_permutations = [[x for x in perm if len(x) > 1] for perm in product(*perm_dict.values())]
    if len(symmetric_permutations) > max_symm_perm:
        import random
        symmetric_permutations = random.sample(symmetric_permutations, max_symm_perm)
    
    def select_points(coords, symmetric_permutations):
        permuted_coords = compute_frame_permutations(coords, symmetric_permutations)
        descriptors = compute_pairwise_distances(permuted_coords)
        compute_distance_func = functools.partial(compute_permutation_invariant_distance, descriptors=descriptors)
        num_points = len(coords)
        return fps(num_points, num_samples, compute_distance_func)

    while len(coords) > chunk_max_dim:
        print(f"Number of points ({len(coords)}) exceeds the maximum chunk size ({chunk_max_dim}). Splitting into chunks.")
        num_chunks = int(np.ceil(len(coords) / chunk_max_dim))
        chunk_size = len(coords) // num_chunks
        chunks_coords = [coords[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        all_selected_points = []
        offset = 0
        for i, chunk_coords in enumerate(chunks_coords):
            print(f"Processing chunk {i+1}/{num_chunks}")
            selected_points = select_points(chunk_coords, symmetric_permutations) + offset
            all_selected_points.extend(selected_points)
            offset += len(chunk_coords)
        
        print(f"Selected {len(all_selected_points)} overall points from all chunks")
        points = points[all_selected_points]
        coords = coords[all_selected_points]
    
    # Perform FPS on the final set of coordinates
    print("Performing FPS on the final set of coordinates")
    final_selected_points = select_points(coords, symmetric_permutations)
    
    return np.sort(points[final_selected_points])

def get_list_filename(h5_filename):
    base, _ = os.path.splitext(h5_filename)
    return base + '.list'

def process_h5_file(h5_filename, n_samples, chunk_max_dim, max_symm_perm):
    coords, atom_types, fullnames, info_dict, extra_data = read_h5_file(h5_filename)
    names = extra_data['symmetry_names_sorted']
    sampled_indices = furthest_point_sampling(n_samples, coords, names, chunk_max_dim=chunk_max_dim, max_symm_perm=max_symm_perm)
    output_filepath = get_list_filename(h5_filename)
    np.savetxt(output_filepath, sampled_indices, fmt='%d')
    update_lower_degree_nmer_sampled_idcs(output_filepath)

def get_root_folder(filename, root_folders):
    for part in reversed(filename.split('/')):
        if part in root_folders:
            return part
    raise ValueError("Root folder not found in filename")

def update_lower_degree_nmer_sampled_idcs(filename):
    print(f"Updating sampled indices list of lower degree nmers for {filename}...")

    # Define the root folders
    root_folders = list(DataDict.FOLDER_NAMES.values())

    # Extract the root folder from the filename
    root_folder = get_root_folder(filename, root_folders)
    sampled_indices = np.loadtxt(filename, dtype=int)
    root_folder2num_monomers = {v: k for k, v in DataDict.FOLDER_NAMES.items()}
    nmers_idcs = os.path.basename(filename).split('.')[-2].split('_')[-root_folder2num_monomers[root_folder]:]

    # Get the base directory
    base_dir = filename.split(root_folder)[0] + root_folder

    # Find subfolders and files
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder in root_folders:
            for file in glob.glob(os.path.join(subfolder_path, '**/*.h5'), recursive=True):
                    file_nmers_idcs = os.path.basename(file).split('.')[-2].split('_')[-root_folder2num_monomers[get_root_folder(file, root_folders)]:]
                    if any(idx in file_nmers_idcs for idx in nmers_idcs):
                        list_file = file.replace('.h5', '.list')
                        list_file_path = os.path.join(subfolder_path, list_file)
                        if os.path.exists(list_file_path):
                            existing_indices = np.loadtxt(list_file_path, dtype=int)
                            updated_indices = np.unique(np.concatenate((existing_indices, sampled_indices)))
                            np.savetxt(list_file_path, updated_indices, fmt='%d')
                        else:
                            np.savetxt(list_file_path, sampled_indices, fmt='%d')

def main():
    """
    Main function to run FPS on H5 files in a specified folder.

    This function parses command-line arguments to get the folder containing H5 files,
    the number of samples to select, the maximum dimension of each chunk, and the maximum
    number of processes to use. It then processes each H5 file in the folder using the
    specified parameters.

    Command-line arguments:
    h5_foldername (str): Path to the folder containing H5 files.
    n_samples (int): Number of samples to select.
    -c, --chunk_max_dim (int, optional): Maximum dimension of each chunk (default: 1e9).
    -s, --max_symm_perm (int, optional): Maximum number of symmetric permutations evaluated (default: 100).
    -p, --max_processes (int, optional): Maximum number of processes to use (default: 1).

    Example usage:
    ```
    nohup dataforge-fps /path/to/h5_folder 1000 -c 5000 -p 4 &
    ```
    """
    parser = argparse.ArgumentParser(description="Run FPS on H5 files in a folder.")
    parser.add_argument("h5_foldername", type=str, help="Path to the folder containing H5 files.")
    parser.add_argument("n_samples", type=int, help="Number of samples to select.")
    parser.add_argument("-c", "--chunk_max_dim", type=int, default=1e9, help="Maximum dimension of each chunk (default: 1e9).")
    parser.add_argument("-s", "--max_symm_perm", type=int, default=100, help="Maximum number of symmetric permutations evaluated (default: 100).")
    parser.add_argument("-p", "--max_processes", type=int, default=64, help="Maximum number of processes to use (default: 64).")

    args = parser.parse_args()

    h5_foldername = args.h5_foldername
    n_samples     = args.n_samples
    chunk_max_dim = args.chunk_max_dim
    max_symm_perm = args.max_symm_perm
    max_processes = args.max_processes

    print(f"Processing H5 files in folder: {h5_foldername}")
    print(f"Number of samples to select: {n_samples}")
    print(f"Maximum dimension of each chunk: {chunk_max_dim}")
    print(f"Maximum number of symmetric permutations evaluated: {max_symm_perm}")
    print(f"Maximum number of processes to use: {max_processes}")

    process_h5_file_func = functools.partial(process_h5_file, n_samples=n_samples, chunk_max_dim=chunk_max_dim, max_symm_perm=max_symm_perm)
    h5_filepaths = glob.glob(os.path.join(h5_foldername, "**/*.h5"), recursive=True)

    if max_processes <= 1:
        for h5_filepath in h5_filepaths:
            process_h5_file_func(h5_filepath)
    else:
        with Pool(processes=min(max_processes, len(h5_filepaths))) as pool:
            pool.map(process_h5_file_func, h5_filepaths)


if __name__ == "__main__":
    main()