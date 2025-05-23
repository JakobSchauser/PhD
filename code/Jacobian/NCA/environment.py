import torch


class Environment:
	def __init__(self, model, positions, k=4, device='cpu'):
		self.positions = torch.tensor(positions, device=device, dtype=torch.float32)
		self.model = model
		self.model.to(device)
		self.num_nodes = self.positions.shape[0]
		self.device = device
		self.k = k
		self.state = None
		self.adjacency = self._make_knn_adjacency(self.positions, k)
		self.edge_info, self.edge_weights = self._adjacency_to_edge_index_and_weights(self.adjacency)

	def _make_knn_adjacency(self, positions, k):
		# Compute pairwise distances
		dists = torch.cdist(positions, positions, p=2)
		# For each node, find k nearest neighbors (excluding self)
		knn_idx = dists.argsort(dim=1)[:, 1:k+1]
		adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
		for i in range(self.num_nodes):
			adj[i, knn_idx[i]] = 1.0
		return adj

	def _adjacency_to_edge_index_and_weights(self, adj):
		# Convert adjacency matrix to edge_index and edge_weight for PyG
		src, dst = torch.nonzero(adj, as_tuple=True,)
		edge_index = torch.stack([src, dst], dim=0,)  # shape [2, num_edges]
		edge_index = edge_index.long()
		edge_index = edge_index.to(self.device)

		edge_weight = adj[src, dst]
		return edge_index.to(self.device), edge_weight.to(self.device)

	def get_adjacency(self):
		return self.adjacency

	def get_positions(self):
		return self.positions

	def train(self, measurements, epochs=100, lr=1e-3, steps_btwn_data = 10):
		"""
		Simple training loop for the NCA model.
		Args:
			model: torch.nn.Module, the NCA or GNN model
			measurements: list measurements of size (num_timepoints, num_nodes)
			steps_btwn_data: number of steps between measurements
			epochs: number of training epochs
			lr: learning rate
		"""
		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		loss_fn = torch.nn.MSELoss()
		losses = []


		for epoch in range(epochs):
			X = torch.zeros_like(measurements[0], device=self.device)
			for targets in measurements:
				optimizer.zero_grad()
				for step in range(steps_btwn_data):
					pred = self.model(X, self.edge_info, self.edge_weights)
					X = pred.detach()
				loss = loss_fn(pred, targets)
				loss.backward()
				optimizer.step()
				losses.append(loss.item())
				# X = targets.detach()
				if epoch % 10 == 0:
					print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
		return losses


	def test_model(self, measurements, steps_btwn_data = 10):
		"""
		Test the model on the given measurements.
		Args:
			model: torch.nn.Module, the NCA or GNN model
			measurements: list measurements of size (num_timepoints, num_nodes)
			steps_btwn_data: number of steps between measurements
		"""
		X = torch.zeros_like(measurements[0], device=self.device)
		predictions = []

		for targets in measurements:
			for step in range(steps_btwn_data):
				pred = self.model(X, self.edge_info, self.edge_weights)
				X = pred.detach()
			predictions.append(pred.detach().numpy())

		return predictions