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
    def __init__(self, input_dims, hidden_dims, output_dims,  biases : bool,aggregation = "add", name = "CustomGNN"):
        super(CustomGNN, self).__init__()
        # check if list
        if type(hidden_dims) is not list:
            hidden_dims = [hidden_dims]
        
        self.name = name
        
        # convolutional layer
        self.input_layer = CustomGraphConv(input_dims*2, hidden_dims[0], aggr=aggregation, bias = biases)

        self.hidden_layers = torch.nn.ModuleList()
        # Linear layers
        for i in range(len(hidden_dims)-1):
            _in = hidden_dims[i]
            _out = hidden_dims[i+1]

            self.hidden_layers.append(Linear(_in, _out, bias = biases))
            # self.hidden_layers.append(CustomGatedPolynomial(_in, _out))


        self.output_layer = Linear(hidden_dims[-1], output_dims, bias = True)



    def forward(self, feature_data, edge_info, edge_weights):

        # First Graph Convolutional layer (message passing)
        x = self.input_layer(feature_data, edge_info, edge_weights)
        x = F.relu(x)


        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)            

        x = self.output_layer(x, )
        
        return x


    def forward_verbose(self, feature_data, edge_info, edge_weights):

        # First Graph Convolutional layer (message passing)
        x = self.input_layer(feature_data, edge_info, edge_weights)
        x = F.relu(x)

        out1 = x.clone()

        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)

        x = self.output_layer(x, )

        return x, out1
    
    def get_weights(self):
        weights = []
        weights.append(self.input_layer.lin_rel.weight)
        # weights.append(self.input_layer.lin_root.weight)

        for hl in self.hidden_layers:
            weights.append(hl.weight)

        weights.append(self.output_layer.weight)

        return weights
    
    def set_weights(self, weights):
        self.input_layer.lin_rel.weight = weights[0]

        for i, hl in enumerate(self.hidden_layers):
            hl.weight = weights[i+1]

        self.output_layer.weight = weights[-1]