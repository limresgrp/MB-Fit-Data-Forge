import os
import glob
import argparse
import numpy as np
from itertools import permutations, product
from sklearn_extra.cluster import KMedoids
from multiprocessing import Pool
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
    all_permuted_coords = []
    for perm in symmetric_permutations:
        permuted_coords = coords.copy()
        for p in perm:
            permuted_coords = apply_permutation(permuted_coords, p)
        all_permuted_coords.append(permuted_coords)
    return np.stack(all_permuted_coords, axis=1)

def compute_pairwise_distances(A):
    _, _, atoms, _ = A.shape
    # Compute the pairwise differences
    diff = A[:, :, :, np.newaxis, :] - A[:, :, np.newaxis, :, :]
    # Compute the squared distances
    dist_squared = np.sum(diff**2, axis=-1)
    # Extract the lower triangular part of the distance matrix (excluding the diagonal)
    i_lower = np.tril_indices(atoms, -1)
    distance_matrix = dist_squared[:, :, i_lower[0], i_lower[1]]
    # Take the square root to get the Euclidean distances
    distance_matrix = np.sqrt(distance_matrix)
    return distance_matrix

def compute_distance_between_frames(coords1, coords2):
    diff = coords1[:, np.newaxis, :] - coords2[:, np.newaxis, :, :]
    # Compute the squared distances
    dist_squared = np.sum(diff**2, axis=-1)
    return dist_squared.min(axis=2).min(axis=1)

def compute_distance(i, descriptors):
    return i, compute_distance_between_frames(descriptors[i, :1], descriptors[i+1:])

def compute_all_distances(coords, names, max_symm_perm=100, max_processes=1):
    print("Generating permutations")
    perm_dict = generate_permutations(names)
    symmetric_permutations = [[x for x in perm if len(x) > 1] for perm in product(*perm_dict.values())]
    if len(symmetric_permutations) > max_symm_perm:
        import random
        symmetric_permutations = random.sample(symmetric_permutations, max_symm_perm)
    print("Computing permuted coordinates")
    permuted_coords = compute_frame_permutations(coords, symmetric_permutations)
    print("Computing symmetry-aware pairwise distances")
    permuted_distances = compute_pairwise_distances(permuted_coords)
    n_frames = coords.shape[0]
    distances = np.zeros((n_frames, n_frames))

    print("Computing frame distances")
    if max_processes > 1:
        with Pool(processes=max_processes) as pool:
            results = pool.starmap(compute_distance, [(i, permuted_distances) for i in range(n_frames)])
    else:
        results = [compute_distance(i, permuted_distances) for i in range(n_frames)]

    for i, dist in results:
        distances[i, i+1:] = dist
        distances[i+1:, i] = dist

    return distances

def furthest_point_sampling(num_samples, coords, names, chunk_max_dim=10000, max_symm_perm=100, max_processes=1, n_clusters=10):
    def fps(M, N):
        num_points = M.shape[0]
        assert N < num_points, "N must be less than the number of points in M"
        
        # Perform K-medoids clustering to get initial medoids
        k = min(N, n_clusters)  # Number of initial points to select using K-medoids
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=0).fit(M)
        medoids = kmedoids.medoid_indices_
        
        # Initialize the list of selected points with the medoids
        selected_points = list(medoids)
        
        # Initialize the array of minimum distances to the selected points
        min_distances = np.min(M[selected_points, :], axis=0)
        
        for _ in range(len(selected_points), N):
            # Find the point that is farthest from the selected points
            farthest_point = np.argmax(min_distances)
            selected_points.append(farthest_point)
            
            # Update the minimum distances to the selected points
            min_distances = np.minimum(min_distances, M[farthest_point, :])
        
        return np.array(selected_points)

    points = np.arange(len(coords))
    while len(coords) > chunk_max_dim:
        print(f"Number of points ({len(coords)}) exceeds the maximum chunk size ({chunk_max_dim}). Splitting into chunks.")
        num_chunks = int(np.ceil(len(coords) / chunk_max_dim))
        chunk_size = len(coords) // num_chunks
        chunks = [coords[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        
        all_selected_points = []
        offset = 0
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{num_chunks}")
            distance_matrix = compute_all_distances(chunk, names, max_symm_perm=max_symm_perm, max_processes=max_processes)
            selected_points = fps(distance_matrix, min(num_samples, len(chunk))) + offset
            all_selected_points.extend(selected_points)
            offset += len(chunk)
        
        print(f"Selected {len(all_selected_points)} overall points from all chunks")
        points = points[all_selected_points]
        coords = coords[all_selected_points]
    
    # Perform FPS on the final set of coordinates
    print("Performing FPS on the final set of coordinates")
    distance_matrix = compute_all_distances(coords, names, max_symm_perm=max_symm_perm, max_processes=max_processes)
    final_selected_points = fps(distance_matrix, num_samples)
    
    return np.sort(points[final_selected_points])

def get_list_filename(h5_filename):
    base, _ = os.path.splitext(h5_filename)
    return base + '.list'

def process_h5_file(h5_filename, n_samples, chunk_max_dim, max_symm_perm, max_processes):
    coords, atom_types, fullnames, info_dict, extra_data = read_h5_file(h5_filename)
    names = extra_data['symmetry_names_sorted']

    sampled_indices = furthest_point_sampling(n_samples, coords, names, chunk_max_dim=chunk_max_dim, max_symm_perm=max_symm_perm, max_processes=max_processes)

    output_filepath = get_list_filename(h5_filename)
    np.savetxt(output_filepath, sampled_indices, fmt='%d')

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
    -c, --chunk_max_dim (int, optional): Maximum dimension of each chunk (default: 5000).
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
    parser.add_argument("-c", "--chunk_max_dim", type=int, default=10000, help="Maximum dimension of each chunk (default: 5000).")
    parser.add_argument("-s", "--max_symm_perm", type=int, default=100, help="Maximum number of symmetric permutations evaluated (default: 100).")
    parser.add_argument("-p", "--max_processes", type=int, default=1, help="Maximum number of processes to use (default: 1).")

    args = parser.parse_args()

    h5_foldername = args.h5_foldername
    n_samples     = args.n_samples
    chunk_max_dim = args.chunk_max_dim
    max_symm_perm = args.max_symm_perm
    max_processes = args.max_processes

    h5_filepaths = glob.glob(os.path.join(h5_foldername, "**/*.h5"), recursive=True)

    for h5_filepath in h5_filepaths:
        print(f"Processing {h5_filepath}...")
        process_h5_file(h5_filepath, n_samples, chunk_max_dim, max_symm_perm, max_processes)

if __name__ == "__main__":
    main()