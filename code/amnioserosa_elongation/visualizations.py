from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt


# use('PyQt5') 

def make_voronoi_plot(all_poss, ax = None, elongated_cells = None, ps = None, qs = None):
    assert all_poss.shape[1] == 2, "Only 2D data is supported"

    if elongated_cells is None:
        elongated_cells = np.zeros(all_poss.shape[0])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    vor = Voronoi(all_poss)

    voronoi_plot_2d(vor, ax)

    ax.scatter(all_poss[:,0], all_poss[:,1], c = elongated_cells)

    if ps is not None:
        # draw ps as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], ps[:, 0], ps[:, 1], color='blue')
    
    if qs is not None:
        # draw qs as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], qs[:, 0], qs[:, 1], color='red')

    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return ax

def make_scatter2D(ax, all_poss, elongated_cells = None, ps = None, qs = None):
    n_dims = all_poss.shape[1]
    assert n_dims == 2, "Only 2D data is supported"

    ax.scatter(all_poss[:,0], all_poss[:,1], c = elongated_cells)

    if ps is not None:
        # draw ps as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], ps[:, 0], ps[:, 1], color='blue')
    if qs is not None:
        # draw qs as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], qs[:, 0], qs[:, 1], color='red')

    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return ax

def make_scatter3D(ax, all_poss, elongated_cells = None, ps = None, qs = None):
    n_dims = all_poss.shape[1]
    assert n_dims == 3, "Only 3D data is supported"

    ax.scatter(all_poss[:,0], all_poss[:,1], all_poss[:,2], c = elongated_cells)

    if ps is not None:
        # draw ps as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], all_poss[:, 2], ps[:, 0], ps[:, 1], ps[:, 2], color='blue')
    if qs is not None:
        # draw qs as arrows
        ax.quiver(all_poss[:, 0], all_poss[:, 1], all_poss[:, 2], qs[:, 0], qs[:, 1], qs[:, 2], color='red')

    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    return ax

def make_scatter_plot(all_poss, ax = None, elongated_cells : np.ndarray = None, ps = None, qs = None):
    n_dims = all_poss.shape[1]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' if n_dims == 3 else None)

    if n_dims == 2:
        return make_scatter2D(ax, all_poss, elongated_cells, ps, qs)
    elif n_dims == 3:
        return make_scatter3D(ax, all_poss, elongated_cells, ps, qs)
    
    raise ValueError("Only 2D or 3D data is supported, of course")
    
