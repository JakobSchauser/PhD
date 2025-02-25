from typing import Tuple, Union

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm
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

class CustomGraphConv(MessagePassing):
	def __init__(
		self,
		in_channels: Union[int, Tuple[int, int]],
		out_channels: int,
		aggr: str = 'add',
		bias: bool = True,
		**kwargs,
	):
		super().__init__(aggr=aggr, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels

		if isinstance(in_channels, int):
			in_channels = (in_channels, in_channels)

		self.lin_rel = Linear(in_channels[0]*2, out_channels, bias=bias)
		# self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
		# self.lin_root = Linear(in_channels[1], out_channels, bias=False)

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		self.lin_rel.reset_parameters()
		# self.lin_root.reset_parameters()


	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
				edge_weight: OptTensor = None, size: Size = None, add_root_weight : bool = True) -> Tensor:

		if isinstance(x, Tensor):
			x = (x, x)

		# propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
		msg = self.propagate(edge_index, x=x, edge_weight=edge_weight,
							 size=size)

		x_r = x[1]  # Root node features.
		# if x_r is not None:
		# 	out = out + self.lin_root(x_r)

		# add the final feature of the root node


		
		msg = torch.cat([x_r, msg], dim=1)


		out = self.lin_rel(msg)
		return out


	def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
		return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

	def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
		return spmm(adj_t, x[0], reduce=self.aggr)
	

class CustomGatedPolynomial(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(CustomGatedPolynomial, self).__init__()

		# check that the output channels is even
		assert out_channels % 2 == 0, "out_channels must be even"

		self.transform = Linear(in_channels, out_channels)

		self.out_channels = out_channels
		self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
		self.bias.data.fill_(0)



	def forward(self, x):
		# x = x + self.lin(x) * torch.sigmoid(self.lin_gate(x))
		transformed = self.transform(x)

		lower_half = transformed[:, :self.out_channels//2]
		upper_half = transformed[:, self.out_channels//2:]

		x = torch.cat([torch.relu(lower_half), torch.exp(upper_half)], dim=1)
		return x + self.bias
	
	def get_weights(self):
		return self.transform.weight,

