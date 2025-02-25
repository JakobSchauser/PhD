from scipy.spatial import KDTree, Voronoi
import numpy as np
from scipy.spatial.distance import euclidean

def find_filtered_voronoi_neighbor_knn_limited_mask(points, n_neighbors=8):
    """
    Compute a Voronoi neighbor mask optimized by limiting tessellation to
    the `n_neighbors` nearest neighbors for each point using KD-Tree,
    and further subset neighbors based on additional criteria.

    Parameters:
    - points (np.ndarray): Array of shape (n_points, 2) representing the 2D points.
    - n_neighbors (int): Number of nearest neighbors to consider for tessellation.

    Returns:
    - neighbor_mask (np.ndarray): Symmetric matrix of shape (n_points, n_points)
      with 1 indicating neighbors and 0 otherwise, after subsetting.
    """
    num_points = len(points)

    tree = KDTree(points)
    dists, indices = tree.query(points, k=n_neighbors + 1)  # +1 to exclude the point itself

    # remove the first column, which is the distance to the point itself
    # indices = indices[:, 1:]

    # Initialize an empty neighbor mask
    neighbor_mask = np.zeros((num_points, num_points), dtype=int)

    # Perform localized Voronoi tessellation for each point
    for i, neighbors in enumerate(indices):
        # Stack the current point with its neighbors
        # local_points = np.vstack([points[i], points[neighbors]])
        local_points = points[neighbors]

        # Compute the Voronoi tessellation for the local neighborhood
        vor = Voronoi(local_points)

        # Extract pairs of neighboring points from the Voronoi ridges
        ridge_pairs = np.array(list(vor.ridge_dict.keys()))
        ridge_pairs = ridge_pairs[(ridge_pairs >= 0).all(axis=1)]  # Keep valid pairs

        # Map local indices back to global indices
        global_ridge_pairs = [[i if idx == 0 else neighbors[idx - 1] for idx in pair] for pair in ridge_pairs]

        # Update the neighbor mask and apply subsetting
        for p1, p2 in global_ridge_pairs:
            # Mark as neighbors initially
            neighbor_mask[p1, p2] = 1
            neighbor_mask[p2, p1] = 1

            # Subset neighbors based on the additional criteria
            A = (points[p1] + points[p2]) / 2.0
            r_ij_half = np.round(0.5 * euclidean(points[p1], points[p2]), 4)

            # Get neighbors of p1 and p2
            neighbors_p1 = np.array([pair[1] if pair[0] == p1 else pair[0] for pair in global_ridge_pairs if p1 in pair])
            neighbors_p2 = np.array([pair[1] if pair[0] == p2 else pair[0] for pair in global_ridge_pairs if p2 in pair])

            # Calculate distances from A to all other neighbors
            all_neighbors = np.concatenate((neighbors_p1, neighbors_p2))
            distances_to_A = np.round(np.linalg.norm(points[all_neighbors] - A, axis=1), 4)

            # Check the condition
            if np.any(distances_to_A < r_ij_half):
                neighbor_mask[p1, p2] = 0
                neighbor_mask[p2, p1] = 0  # The relationship is symmetric


    # remove the diagonal
    np.fill_diagonal(neighbor_mask, 0)

    # assert symmetric
    # assert np.allclose(neighbor_mask, neighbor_mask.T)


    # make into a [N_cells, N_neighbors] matrix

    n_nbs_for_i = np.sum(neighbor_mask, axis=1)
    nbs = -np.ones((num_points, np.max(n_nbs_for_i)), dtype=int)

    for i in range(neighbor_mask.shape[0]):
        wh = np.where(neighbor_mask[i])[0]
        nbs[i, :n_nbs_for_i[i]]  = wh

    dists = np.zeros((num_points, np.max(n_nbs_for_i)), dtype=float)
    for i in range(num_points):
        dists[i, :n_nbs_for_i[i]] = np.linalg.norm(points[nbs[i, :n_nbs_for_i[i]]] - points[i], axis=1)

    return nbs, dists

    # return neighbor_mask


def from_nb_list_to_adj_matrix(nbs):
    n_cells = nbs.shape[0]
    adj_matrix = np.zeros((n_cells, n_cells), dtype=bool)
    for i in range(n_cells):
        adj_matrix[i, nbs[i, nbs[i] != -1]] = True
    return adj_matrix