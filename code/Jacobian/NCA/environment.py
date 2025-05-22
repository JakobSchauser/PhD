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

    def _make_knn_adjacency(self, positions, k):
        # Compute pairwise distances
        dists = torch.cdist(positions, positions, p=2)
        # For each node, find k nearest neighbors (excluding self)
        knn_idx = dists.argsort(dim=1)[:, 1:k+1]
        adj = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
        for i in range(self.num_nodes):
            adj[i, knn_idx[i]] = 1.0
        return adj


    def get_adjacency(self):
        return self.adjacency

    def get_positions(self):
        return self.positions

    def train(self, measurements, epochs=100, lr=1e-3, steps_btwn_data = 10):
        """
        Simple training loop for the NCA model.
        Args:
            model: torch.nn.Module, the NCA or GNN model
            measurements: list of (node_idx, target_value) tuples or array
            epochs: number of training epochs
            lr: learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        losses = []

        state = torch.zeros_like(measurements[0], device=self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass: model(state, adjacency) or similar
            for targets in measurements:
                for step in range(steps_btwn_data):
                    pred = self.model(pred, self.adjacency)
                loss = loss_fn(pred, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        return losses

