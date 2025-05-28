# graph neural network 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


import torch
from torch import tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, GraphConv

import rootutils

from nb_functions import find_filtered_voronoi_neighbor_knn_limited_mask
from custon_nca import CustomGNN




class Environment():
	def __init__(self, model : CustomGNN,  data, positions, lr : float, weight_gain : float, diversity_gain :float, steps_per_data_point :int= 1):
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

		self.model : CustomGNN = model
		self.previous_model_weights = None
		self.optimizer = None

		self.early_stop_count = 0

		self.lr = lr
		self.weight_gain = weight_gain
		self.diversity_gain = diversity_gain
		self.steps_per_data_point = steps_per_data_point
		self.set_model(model)

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
		

		return base_loss 

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
		# print(     X.shape,            edges.shape,   edge_weights.shape, border_mask.shape)
		# torch.Size([N_cells]) torch.Size([2, 5982]) torch.Size([5982]) torch.Size([N_cells])
		if X.dim() == 1:
			X = X.unsqueeze(1)
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
		



	def train(self, epochs,):
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


				n_steps = len(yy) - 1

				X = torch.zeros_like(yy[0])

				for i in range(n_steps):
					self.optimizer.zero_grad()
					loss = torch.tensor(0.0)

					target = yy[(i+1)] 

					# X, target = self.transformation(X, target)

					# GT_out = self.call_own_model(GT, edges, edge_weights, border_mask)
					for j in range(self.steps_per_data_point):
						out = self.call_own_model(X, edges, edge_weights, border_mask)
						X = out.detach()

					l_loss = self.loss_fn(out, target)# + self.loss_fn(GT_out, target)
					
					loss += l_loss 

					loss += self.loss_addition_cutoff()  # add the l1 loss

					loss.backward()

					self.optimizer.step()
					avg_loss += loss.item()

					X = target.detach()  # reset X to target for next step
		
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
		N_steps = int(len(y_test)) - 1
		for i in range(N_steps):
			for j in range(self.steps_per_data_point):
				out = self.call_own_model(X, edges, edge_weights, border)
				X = out.detach()

			target = y_test[(i+1)]


			off = torch.linalg.norm(torch.abs(out - target), axis=1)

			quality += torch.mean(off)
			

		quality /= N_steps


		self.model.train()

		return quality.item() 

	
	def test_quality(self, show=0):

		data_i = np.random.randint(0, len(self.ys)) 

		poss = self.positions[data_i]
		y_val = self.ys[data_i]

		edges = self.edges[data_i]
		edge_weights = self.edge_weights[data_i]
		border_mask = self.border_mask[data_i]

		X = torch.zeros_like(y_val[0], dtype=torch.float32)

		quality = 0.
		for i in range(int(len(y_val)-1)):

			for j in range(self.steps_per_data_point):
				out = self.call_own_model(X, edges, edge_weights, border_mask)
				X = out.detach()

			target = y_val[(i+1)]
			plottarget = y_val[(i+1)]

			off = np.linalg.norm(np.abs(out.detach().numpy() - target.detach().numpy()), axis=1)

			quality += np.mean(off)

			if show > 0 and(i+1)% (int(len(y_val)/show)) == 0:
				plotshow = X.detach().numpy()
				if len(plottarget.shape) == 1:
					plottarget = plottarget.unsqueeze(1)
				if len(plotshow.shape) == 1:
					plotshow = plotshow.unsqueeze(1)

				shp = plotshow.shape[1]
				fig, axs = plt.subplots(1,2*shp, figsize=(4*shp,2), sharey=True, constrained_layout=True)
				for j in range(shp):
					axs[j*shp+0].scatter(poss[:,0], poss[:,1], c=plotshow[:,j], s = 5.)
					axs[j*shp+1].scatter(poss[:,0], poss[:,1], c=plottarget[:,j], s = 5.)
				plt.show()
			# X = torch.tensor(out, dtype=torch.float32).squeeze(1)

		return quality