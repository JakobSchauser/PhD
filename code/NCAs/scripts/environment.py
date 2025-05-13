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
    def __init__(self, data, positions, lr : float, weight_gain : float, diversity_gain :float, stepsize : int, iterative_training : bool, ):
        # load the data

        self.positions = [p.copy() for p in positions]

        self.ys = []       

        
        self.edges = []
        self.edge_weights = []
        self.border_mask = []

        for d, pos in zip(data, self.positions):
            self.ys.append(torch.tensor(d.copy(), dtype=torch.float32))
            edge, edge_weights, border_mask = self.get_edges(pos)
            self.edges.append(edge)
            self.edge_weights.append(edge_weights)
            self.border_mask.append(border_mask)

        self.model = None
        self.previous_model_weights = None
        self.optimizer = None

        self.early_stop_count = 0

        self.lr = lr
        self.weight_gain = weight_gain
        self.diversity_gain = diversity_gain
        self.stepsize = stepsize
        self.iterative_training = iterative_training


    def loss_addition(self):
        if self.weight_gain == 0.:
            return 0.
    
        l1_weights = torch.stack([wh.abs().sqrt().sum() for wh in self.model.get_weights()]).mean()


        return (l1_weights)*self.weight_gain

    def set_diversity_gain(self, diversity_gain):
        self.diversity_gain = diversity_gain

    def set_weight_gain(self, weight_gain):
        self.weight_gain = weight_gain

    def loss_addition_cutoff(self):
        if self.weight_gain == 0.:
            return 0.
    
        threshold = 1e-4

        add = 0.
        for wh in self.model.get_weights():
            lay = wh.abs() - threshold
            add += torch.relu(lay).sum()

        return add*self.weight_gain

    def loss_fn(self, out, target):
        base_loss = F.mse_loss(out, target)
        

        return base_loss * 1000.

    @staticmethod
    def get_edges(positions):

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
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

        # create edge data
        edges = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long).t().contiguous().T
        edge_weights = torch.tensor(adj_matrix[adj_matrix > 0], dtype=torch.float32)

        # make border array
        border_mask = Environment.find_border_mask(adj_matrix, positions)
        
        return edges, edge_weights, torch.tensor(border_mask)

    @staticmethod
    def find_border_mask(adjacency_matrix, positions):
        # return (adjacency_matrix>0.).sum(axis = 0) <= 4
    
        boundary_mask = np.zeros(positions.shape[0], dtype=bool)

        for i in range(positions.shape[0]):
            if sum(adjacency_matrix[i]) == 0:
                boundary_mask[i] = True
                continue
                
            nb_poss =  positions[adjacency_matrix[i]>0.]
            vecs_to_nbs = nb_poss - positions[i]

            angles = np.arctan2(vecs_to_nbs[:, 1], vecs_to_nbs[:, 0])

            sorted_angles = np.sort(angles)

            # find the largest gap, taking into account the wrap around at 2pi
            diffs = np.diff(sorted_angles)
            diffs = np.concatenate([diffs, [2*np.pi + sorted_angles[0] - sorted_angles[-1]]])
            boundary_mask[i] = diffs.max()>np.pi*0.9


        return boundary_mask
    

    def set_model(self, model):

        # from intel_npu_acceleration_library.compiler import CompilerConfig
        # compiler_conf = CompilerConfig(dtype=torch.float32, training=True)
        # compiled_model = intel_npu_acceleration_library.compile(model, compiler_conf)
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def set_previous_model(self, model):
        weights = model.get_weights()

        detached_weights = [w.detach() for w in weights]
        self.previous_model_weights = detached_weights


    def call_own_model(self, X, edges, edge_weights, border_mask):
        
        return self.call_model(self.model, X, edges, edge_weights, border_mask)


    @staticmethod
    def call_model(model, X, edges, edge_weights, border_mask):
        # print(X.shape, edges.shape, edge_weights.shape, border_mask.shape)
        # torch.Size([N_cells]) torch.Size([2, 5982]) torch.Size([5982]) torch.Size([N_cells])
        X = torch.cat((X, border_mask.unsqueeze(1)), dim = 1)

        
        return model(X, edges, edge_weights)


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


                n_steps = int(len(yy)//self.stepsize) - 1

                X = torch.zeros_like(yy[0])

                for i in range(n_steps):
                    self.optimizer.zero_grad()
                    loss = torch.tensor(0.0)

                    target = yy[(i+1)*self.stepsize] 

                    # X, target = self.transformation(X, target)

                    # GT_out = self.call_own_model(GT, edges, edge_weights, border_mask)

                    out = self.call_own_model(X, edges, edge_weights, border_mask)

                    l_loss = self.loss_fn(out, target)# + self.loss_fn(GT_out, target)
                    loss += l_loss / n_steps 

                    # GT = target.detach()
                    if self.iterative_training:
                        X = out.detach()
                    else:
                        X = target.detach()

                    # loss += self.loss_addition() / n_steps
                    loss += self.loss_addition_cutoff() / n_steps
                    addddd = 0.
                    if self.previous_model_weights is not None:
                        addddd = self.common_loss_function() / n_steps * self.diversity_gain
                        loss += addddd

                    loss.backward()

                    self.optimizer.step()
                    avg_loss += loss.item()
        
            avg_loss /= len(self.ys)

            avg_loss = avg_loss.item() * 1000.

            if epoch % 10 == 0:
                print(epoch)
                test_accuracy = self.get_accuracy()

                # test_loss = self.test()
                # if self.check_early_stop(avg_loss, test_loss):
                    # print('Early stoppping')
                l1_weights = torch.sum(torch.tensor([F.mse_loss(wh, torch.zeros_like(wh)) for wh in self.model.get_weights()]))


                print(f"{self.model.name} | {epoch/epochs:.3} loss:", avg_loss, "| accuracy:", test_accuracy, "| l1 weights:", l1_weights.item(), ("" if self.previous_model_weights is None else  f"| diversity: {self.common_loss_function().item()}"))


    
    def get_accuracy(self):
        yi = np.random.randint(0, len(self.ys))

        y_test = self.ys[yi]

        edges = self.edges[yi]
        edge_weights = self.edge_weights[yi]
        border = self.border_mask[yi]
        X = torch.zeros_like(y_test[0])

        self.model.eval()

        quality = torch.tensor(0.0)
        N_steps = int(len(y_test)//self.stepsize) - 1
        for i in range(N_steps):
            out = Environment.call_model(self.model, X, edges, edge_weights, border)

            target = y_test[(i+1)*self.stepsize]


            off = torch.linalg.norm(torch.abs(out - target), axis=1)

            quality += torch.mean(off)
            
            X = out


        quality /= N_steps


        self.model.train()

        return quality.item() 


    def common_loss_function(self,):
        weights1 = self.model.get_weights()
        weights2 = self.previous_model_weights

        return self.get_diversity_symmetric(weights1, weights2)


    @staticmethod
    def get_diversity(weights1, weights2):
        sums = 0.
        for w1, w2 in zip(weights1, weights2):
            assert w1.shape == w2.shape, "Weights are not the same shape"
            w1 = w1.T
            w2 = w2.T

            a_b_diffs = torch.linalg.norm(w1[None, :, :] - w2[:, None, :], dim=2)

            a_norms = torch.linalg.norm(w1, dim=1)
            b_norms = torch.linalg.norm(w2, dim=1)
            layer = torch.prod(a_b_diffs, dim=1) / (a_norms + b_norms)
            sums += torch.sum(layer)
            break

        return -sums
    


    @staticmethod
    def get_diversity_symmetric(weights1, weights2):
        sums = 0.
        for w1, w2 in zip(weights1, weights2):
            assert w1.shape == w2.shape, "Weights are not the same shape"
            w1 = w1.T
            w2 = w2.T

            a_b_diffs = torch.linalg.norm(w1[None, :, :] - w2[:, None, :], dim=2)

            # a_norms = torch.linalg.norm(w1, dim=1)
            # b_norms = torch.linalg.norm(w2, dim=1)

            layer1 = torch.prod(a_b_diffs , dim=0)# / (a_norms + b_norms)
            layer2 = torch.prod(a_b_diffs , dim=1)# / (a_norms + b_norms)
            sums += torch.sum(layer1) + torch.sum(layer2)
            break

        return -sums
    