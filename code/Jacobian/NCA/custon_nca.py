
from typing import Tuple, Union


import torch

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm

import torch.nn.functional as F

class CustomGNN(torch.nn.Module):
	def __init__(self, input_dims, hidden_dims, output_dims,  biases : bool, aggregation = "add", name = "CustomGNN"):
		super(CustomGNN, self).__init__()

		# check if list
		if type(hidden_dims) is not list:
			hidden_dims = [hidden_dims]
		
		self.name = name
		
		# convolutional layer
		self.input_layer = CustomGraphConv(input_dims*2 + 1, hidden_dims[0], aggr=aggregation, bias = biases)

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


	def get_weights(self):
		weights = []
		weights.append(self.input_layer.lin_rel.weight)

		for hl in self.hidden_layers:
			weights.append(hl.weight)

		weights.append(self.output_layer.weight)

		return weights
	


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

		self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
		# self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
		# self.lin_root = Linear(in_channels[1], out_channels, bias=False)

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()
		self.lin_rel.reset_parameters()
		# self.lin_root.reset_parameters()


	def forward(self, x: Tensor, edge_index: Adj,
				edge_weight: OptTensor = None, size: Size = None, add_root_weight : bool = True) -> Tensor:
		x = x.unsqueeze(1) if x.dim() == 1 else x
				  

		msg = self.propagate(edge_index, x=x, edge_weight=edge_weight,
							 size=size)

		node_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).float() / x.size(0)


		out_msg = torch.cat((x, msg, node_indices), dim=1)
		# out_msg = torch.cat((x, msg), dim=1)


		out = self.lin_rel(out_msg)
		
		return out


	def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
		return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

	def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
		return spmm(adj_t, x[0], reduce=self.aggr)
	