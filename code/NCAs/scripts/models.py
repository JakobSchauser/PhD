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

from scripts.layers import CustomGraphConv


class CustomGNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(CustomGNN, self).__init__()
        # check if list
        if type(hidden_dims) is not list:
            hidden_dims = [hidden_dims]
        
        # convolutional layer
        self.input_layer = CustomGraphConv(input_dims, hidden_dims[0])

        self.hidden_layers = torch.nn.ModuleList()
        # Linear layers
        for i in range(len(hidden_dims)-1):
            _in = hidden_dims[i]
            _out = hidden_dims[i+1]

            self.hidden_layers.append(Linear(_in, _out))
            # self.hidden_layers.append(CustomGatedPolynomial(_in, _out))


        self.output_layer = Linear(hidden_dims[-1], output_dims)

        no_img_sqrt = lambda x: torch.sqrt(torch.abs(x))

        self.possible_functions = [lambda x: x, no_img_sqrt, torch.square, torch.exp]*2

    def forward(self, feature_data, edge_info, edge_weights):

        # First Graph Convolutional layer (message passing)
        x = self.input_layer(feature_data, edge_info, edge_weights)
        x = F.relu(x)


        # Second GCN layer
        

        for layer in self.hidden_layers:
            # for i, (f, xx) in enumerate(zip(self.possible_functions, x.T)):
            #     assert not torch.isnan(f(xx)).any(), f"Function {i} produced NaNs!"

            # for each output feature apply a function
            # x =  torch.cat([f(xx).unsqueeze(0) for f, xx in zip(self.possible_functions, x.T)]).T
            # x = F.relu(x)
            
            # square
            # C = 100.
            # x = torch.clamp(x, -C, C)  # Choose C ~1-10 to prevent extreme values

            # lower_half = x[:, :x.shape[1]//2]
            # upper_half = x[:, x.shape[1]//2:]

            # x = torch.cat([lower_half, F.softplus(upper_half)], dim=1)

            x = layer(x)
            x = F.softplus(x)
            

        x = self.output_layer(x, )
        # x = self.layer3(x, edge_info, edge_weights)
        # x = torch.tanh(x)
        
        return x
    

    
    def get_weights(self):
        weights = []
        weights.append(self.input_layer.lin_rel.weight)
        # weights.append(self.input_layer.lin_root.weight)

        for hl in self.hidden_layers:
            weights.append(hl.weight)
            # for w in hl.get_weights():
            #     weights.append(w)

        weights.append(self.output_layer.weight)

        return weights
    
