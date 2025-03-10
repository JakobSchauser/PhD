import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scripts.environment import Environment
import torch

def prune(model, threshold=0.01):

    weights = [w.detach().numpy() for w in model.get_weights()]

    print(weights)
    sizes = [w.shape[0] for w in weights]
    sizes = [weights[0].shape[1]] + sizes

    print(sizes)

    G = nx.complete_multipartite_graph(*sizes)
    pos = nx.multipartite_layout(G)

    DG = G.to_directed()

    # remove all edges
    DG.remove_edges_from(list(DG.edges()))

    # find nodes in each layer
    nodes_in_subsets = [[] for _ in range(len(sizes))]
    for node in DG.nodes(data=True):
        subset = node[1]['subset']
        nodes_in_subsets[subset].append(node[0])

    prune_masks = []
    # remove edges with small weights
    for weight_matrix_i in range(len(weights)):
        weight_matrix = weights[weight_matrix_i]

        weight_matrix_iii = weight_matrix_i #+ (-1 if weight_matrix_i != 0 else 0) # beacause first two are for same layer

        edges_to_add = []
        mean_mag = np.mean(np.abs(weight_matrix))

        to_remove_mask = np.ones_like(weight_matrix, dtype=bool)
        for i, w in enumerate(weight_matrix):
            mean_mag_row = np.mean(np.abs(w))
            for j, w_ in enumerate(w):
                sens = threshold
                if np.abs(w_) > sens*mean_mag_row and np.abs(w_) > sens*mean_mag:
                    edges_to_add.append((nodes_in_subsets[weight_matrix_iii][j], nodes_in_subsets[weight_matrix_iii+1][i], {"weight":w_}))
                    to_remove_mask[i, j] = False

        DG.add_edges_from(edges_to_add)

        prune_masks.append(to_remove_mask)
        plt.imshow(weight_matrix)
        plt.colorbar()
        plt.scatter(np.where(to_remove_mask)[1], np.where(to_remove_mask)[0], c='r')
        plt.show()


    # # remove nodes with no outgoing edges or incoming edges
    node_colors = []
    for node in DG.nodes(data=True):
        col = "lightblue"
        if len(list(DG.in_edges(node[0]))) == 0 and not node[1]['subset'] == 0:
            col = 'r'
        elif len(list(DG.out_edges(node[0]))) == 0 and not node[1]['subset'] == len(sizes)-1:
            col = 'r'

        node_colors.append(col)

    # color the edges based on the weight matrix


    labels = {}

    for node in nodes_in_subsets[0]:
        n = node - nodes_in_subsets[0][0]
        names  = ["SMAD", "ERK", "Border", "nb_SMAD", "nb_ERK", "nb_Border"]
        # names  = ["BMP", "Border", "nb_BMP", "nb_Border"]
        # names  = ["SMAD", "Border", "nb SMAD", "nb Border", "nb_ERK", "nb_Border"]
        labels[node] = names[n]

    for node in nodes_in_subsets[-1]:
        n = node - nodes_in_subsets[-1][0]
        names = ["SMAD", "ERK"]
        # names = ["BMP"]
        labels[node] = names[n]

    for i in range(1, len(sizes)-1):
        for node in nodes_in_subsets[i]:
            n = node - nodes_in_subsets[i][0]
            labels[node] = f"HL{i} {n}"
            

            
    # discrete colormap
    cmap = plt.cm.coolwarm
    # Create a discrete colormap
    cmap = plt.cm.get_cmap('coolwarm', 10)


    vmin, vmax = -.5, .5


    edges = DG.edges(data=True)
    plotweights = [edge[2]['weight'] for edge in edges]
    fig, ax = plt.subplots()
    nx.draw(DG, pos, with_labels=True, node_color=node_colors, edge_color=plotweights, edge_cmap=cmap, labels=labels, width=2, edge_vmin=vmin, edge_vmax=vmax)
    nx.draw_networkx_edge_labels(DG, pos, edge_labels={(e[0], e[1]): f"{e[2]['weight']:.3f}" for e in edges})

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    # fig.colorbar(sm, ax=ax)

    plt.show()

    return prune_masks


def get_quality(model, data, positions, stepsize, show=0):
    assert len(data) == len(positions)
    model.eval()


    ii = np.random.randint(0, len(data)) 

    poss = positions[ii]
    y_val = data[ii]

    edges, edge_weights, border = Environment.get_edges(poss)

    X = torch.zeros((poss.shape[0],2), dtype=torch.float32)
    # X = y_val[100]

    quality = 0.
    for i in range(int(len(y_val)/stepsize)):
        out = Environment.call_model(model, X, edges, edge_weights, border)


        target = y_val[(i+1)*stepsize]

        off = np.linalg.norm(np.abs(out.detach().numpy() - target), axis=1)

        quality += np.mean(off)
        X = out

        if show > 0 and(i+1)% (int(len(y_val)/stepsize/show)) == 0:
            fig, axs = plt.subplots(1,4, figsize=(8,2), sharey=True, constrained_layout=True)
            axs[0].scatter(poss[:,0], poss[:,1], c=out.detach().numpy()[:,0], s = 5.)
            axs[0].set_title("SMAD Prediction")
            axs[1].scatter(poss[:,0], poss[:,1], c=target[:,0], s = 5.)
            axs[1].set_title("SMAD Target")
            axs[2].scatter(poss[:,0], poss[:,1], c=out.detach().numpy()[:,1], s = 5.)
            axs[2].set_title("ERK Prediction")
            axs[3].scatter(poss[:,0], poss[:,1], c=target[:,1], s = 5.)
            axs[3].set_title("ERK Target")
            plt.show()
        # X = torch.tensor(out, dtype=torch.float32).squeeze(1)

    return quality