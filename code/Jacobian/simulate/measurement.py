import numpy as np

def make_measurements(data, coords, n=1):
    """
    Extracts every n-th timepoint for the given coordinates from the simulation data.
    Args:
        data: np.ndarray, shape (timesteps, x, y)
        coords: list of (x, y) tuples
        n: int, step size for timepoints
    Returns:
        measurements: np.ndarray, shape (num_timepoints, num_coords)
    """
    time_indices = np.arange(0, data.shape[0], n)
    measurements = np.array([
        [data[t, x, y] for (x, y) in coords] for t in time_indices
    ])
    return measurements
