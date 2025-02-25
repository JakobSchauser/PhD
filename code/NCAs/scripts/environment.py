# graph neural network 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


import torch
from torch import tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, GraphConv

import rootutils
root = rootutils.setup_root(".", dotenv=True, pythonpath=True,  indicator =  [".project-root"], cwd = True)

from scripts.nb_functions import find_filtered_voronoi_neighbor_knn_limited_mask





class Environment():
    def __init__(self, data, positions):
        # load the data

        self.positions = positions

        self.ys = []       
        self.y_test = [] 

        
        self.edges = []
        self.edge_weights = []
        self.border_mask = []

        for d, pos in zip(data, self.positions):
            self.ys.append(torch.tensor(d, dtype=torch.float32))
            edge, edge_weights, border_mask = self.get_edges(pos)
            self.edges.append(edge)
            self.edge_weights.append(edge_weights)
            self.border_mask.append(border_mask)

        self.model = None
        self.optimizer = None

        self.early_stop_count = 0

        

    def loss_fn(self, out, target):
        base_loss = F.mse_loss(out, target)
        
        l1_weights = torch.stack([wh.abs().sqrt().sum() for wh in self.model.get_weights()]).mean()

        # flatten

        addition = 1e-6*l1_weights
        # addition = 0.


        return base_loss + addition
    

    def get_edges(self, positions):

        # create a graph with 1000 nodes
        # create a KD tree for fast nearest neighbor search
        # tree = KDTree(positions)
        # dists, indices = tree.query(positions, k=10)

        # indices[dists > 4.5] = -1

        indices, dists = find_filtered_voronoi_neighbor_knn_limited_mask(positions, 8,)
        print(indices.shape, dists.shape)
        # create adjacency matrix
        adj_matrix = np.zeros((indices.shape[0], indices.shape[0]))
        for i in range(indices.shape[0]):
            for ji, j in enumerate(indices[i]):
                if j == -1:
                    continue
                adj_matrix[i, j] = dists[i, ji]
                adj_matrix[j, i] = dists[i, ji]

        # create edge data
        edges = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long).t().contiguous().T
        edge_weights = torch.tensor(adj_matrix[adj_matrix > 0], dtype=torch.float32)

        # make border array if less than 3 nbs
        border_mask = (adj_matrix>0.).sum(axis = 0) <= 4
        
        return edges, edge_weights, torch.tensor(border_mask)

    def set_model(self, model):

        # from intel_npu_acceleration_library.compiler import CompilerConfig
        # compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
        # compiled_model = intel_npu_acceleration_library.compile(model, compiler_conf)
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def call_model(self, X, edges, edge_weights, border_mask):
        X = torch.cat((X, border_mask.unsqueeze(1)), dim = 1)
        
        return self.model(X, edges, edge_weights)


    

    def check_early_stop(self, avg_loss, test_loss):
        if test_loss < avg_loss:
            self.early_stop_count = 0
            return False

        if self.early_stop_count >= 3:
            return True
        
        self.early_stop_count += 1
        return False
        



    def train(self, epochs):
        assert self.model is not None, "Model is not initialized"

        # train the model
        self.model.train()
        for epoch in range(1,epochs+1):
            avg_loss = torch.tensor(0.0)
            for data_i in range(len(self.ys)):
                yy = self.ys[data_i]
                edges = self.edges[data_i]
                edge_weights = self.edge_weights[data_i]
                border_mask = self.border_mask[data_i]


                n_steps = 120
                X = yy[0]
                self.optimizer.zero_grad()

                loss = torch.tensor(0.0)

                for i in range(n_steps):

                    target = yy[(i+1)*5]

                    # X, target = self.transformation(X, target)

                    out = self.call_model(X, edges, edge_weights, border_mask)

                    l_loss = self.loss_fn(out, target) 

                    loss += l_loss

                    X = out.detach()

                loss.backward()

                self.optimizer.step()
                avg_loss += loss.item()
            
            avg_loss /= n_steps

            avg_loss = avg_loss.item()

            if epoch % 5 == 0:
                print(epoch)
                print(f"{epoch/epochs:.3} loss:", avg_loss)
                # test_loss = self.test()
                # if self.check_early_stop(avg_loss, test_loss):
                    # print('Early stoppping')
                l1_weights = torch.sum(torch.tensor([F.mse_loss(wh, torch.zeros_like(wh)) for wh in self.model.get_weights()]))
                print(l1_weights.item())
                    # break

 
