import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scripts.environment import Environment
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def show_graph(weights, prune_masks=None):
    sizes = [w.shape[0] for w in weights]
    sizes = [weights[0].shape[1]] + sizes

    G = nx.complete_multipartite_graph(*sizes)
    pos = nx.multipartite_layout(G)
    DG = G.to_directed()
    DG.remove_edges_from(list(DG.edges()))

    nodes_in_subsets = [[] for _ in range(len(sizes))]
    for node in DG.nodes(data=True):
        subset = node[1]['subset']
        nodes_in_subsets[subset].append(node[0])

    for weight_matrix_i, weight_matrix in enumerate(weights):
        mask = prune_masks[weight_matrix_i] if prune_masks else np.zeros_like(weight_matrix, dtype=bool)
        edges_to_add = []
        for i, w in enumerate(weight_matrix):
            for j, w_ in enumerate(w):
                if not mask[i, j]:
                    edges_to_add.append((nodes_in_subsets[weight_matrix_i][j], 
                                         nodes_in_subsets[weight_matrix_i+1][i], 
                                         {"weight": w_}))
        DG.add_edges_from(edges_to_add)

    node_colors = ["r" if len(list(DG.in_edges(node[0]))) == 0 and not node[1]['subset'] == 0 
                   else "lightblue" for node in DG.nodes(data=True)]

    labels = {}
    for node in nodes_in_subsets[0]:
        labels[node] = ["SMAD", "ERK", "Border", "nb_SMAD", "nb_ERK", "nb_Border"][node - nodes_in_subsets[0][0]]
    for node in nodes_in_subsets[-1]:
        labels[node] = ["SMAD", "ERK"][node - nodes_in_subsets[-1][0]]
    for i in range(1, len(sizes)-1):
        for node in nodes_in_subsets[i]:
            labels[node] = f"HL{i} {node - nodes_in_subsets[i][0]}"

    cmap = plt.cm.get_cmap('coolwarm', 10)
    edges = DG.edges(data=True)
    plotweights = [edge[2]['weight'] for edge in edges]
    fig, ax = plt.subplots()
    nx.draw(DG, pos, with_labels=True, node_color=node_colors, edge_color=plotweights, edge_cmap=cmap, labels=labels, width=2)
    nx.draw_networkx_edge_labels(DG, pos, edge_labels={(e[0], e[1]): f"{e[2]['weight']:.3f}" for e in edges})
    plt.show()

def prune(model, threshold=0.01):
    weights = [w.detach().numpy() for w in model.get_weights()]
    sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]

    prune_masks = []
    for weight_matrix in weights:
        mean_mag = np.mean(np.abs(weight_matrix))
        to_remove_mask = np.ones_like(weight_matrix, dtype=bool)
        for i, w in enumerate(weight_matrix):
            mean_mag_row = np.mean(np.abs(w))
            for j, w_ in enumerate(w):
                if np.abs(w_) > threshold * mean_mag_row and np.abs(w_) > threshold * mean_mag:
                    to_remove_mask[i, j] = False
        prune_masks.append(to_remove_mask)
        plt.imshow(weight_matrix)
        plt.colorbar()
        plt.scatter(np.where(to_remove_mask)[1], np.where(to_remove_mask)[0], c='r')
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