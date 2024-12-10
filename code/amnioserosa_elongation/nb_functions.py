import numpy as np
from scipy.spatial import KDTree
# import ball_tree
# from sklearn.neighbors import BallTree



def distance_cutoff(points, cutoff):
    """
    Find all pairs of points that are within a certain distance of each other.
    """
    tree = KDTree(points)
    indices = tree.query_pairs(r=cutoff)
    return indices

def knn(points, k):
    """
    Find the k nearest neighbors for each point.
    """
    tree = KDTree(points)
    _, indices = tree.query(points, k=k)
    return indices


def cutoff_and_knn(points, cutoff, k):
    """
    Find all up to k points are are witin a certain distance of each other
    """
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k)
    
    indices[distances > cutoff] = -1

    return indices

def nearest_neighbors_with_occlusion_kdtree_smart(points, k):
    """
    Find up to k nearest visible neighbors for each point, considering occlusion, using KDTree for efficient neighbor searches.
    """
    tree = KDTree(points)
    n = len(points)
    visible_neighbors = [[] for _ in range(n)]

    for i in range(n):
        p1 = points[i]
        # Query all points sorted by distance
        distances, indices = tree.query(p1, k=n)

        # Matrix of line segments (p1, p2) and checks for occlusion
        neighbors = []
        checked = set()  # To avoid redundant calculations

        for j in indices:
            if i == j or j in checked:
                continue
            p2 = points[j]
            # Vectorized occlusion check against all other points
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                continue

            relative_positions = points - p1
            t_values = np.einsum("ij,j->i", relative_positions, line_vec) / line_len_sq
            is_between = (0 < t_values) & (t_values < 1)
            projections = p1 + t_values[:, None] * line_vec
            distances_to_line = np.linalg.norm(points - projections, axis=1)
            occluding = np.any((distances_to_line < 1e-8) & is_between & (np.arange(n) != i) & (np.arange(n) != j))

            if not occluding:
                neighbors.append(j)
                checked.add(j)
                if len(neighbors) == k:
                    break

        visible_neighbors[i] = neighbors

    return visible_neighbors

