import numpy as np
import os

class DiffusionSimulator:
    def __init__(self, grid_size=(100, 100), dt=0.1, diff_a=0.5, diff_b=1.0, steps=1000, a_external=10.0, b_external=5.0):
        self.grid_size = grid_size
        self.dt = dt
        self.diff_a = diff_a
        self.diff_b = diff_b
        self.steps = steps
        self.a_external = a_external
        self.b_external = b_external
        self.a = np.zeros(grid_size)
        self.b = np.zeros(grid_size)
        self.reset()

    def reset(self):
        self.a = np.zeros(self.grid_size)
        self.b = np.zeros(self.grid_size)
        self.a[:, 0] = self.a_external
        self.b[:, 0] = self.b_external

    @staticmethod
    def laplacian(Z):
        return (
            -4 * Z
            + np.roll(Z, 1, axis=0)
            + np.roll(Z, -1, axis=0)
            + np.roll(Z, 1, axis=1)
            + np.roll(Z, -1, axis=1)
        )

    def update(self):
        a_new = self.a + self.diff_a * self.laplacian(self.a) * self.dt
        b_new = self.b + self.diff_b * self.laplacian(self.b) * self.dt
        self.a = a_new
        self.b = b_new
        # Enforce boundary condition
        self.a[:, 0] = self.a_external
        self.b[:, 0] = self.b_external

    def run(self, steps=None, save_path=None, save_interval=10):
        if steps is None:
            steps = self.steps
        a_hist = []
        b_hist = []
        for i in range(steps):
            self.update()
            if i % save_interval == 0:
                a_hist.append(self.a.copy())
                b_hist.append(self.b.copy())
        a_hist = np.stack(a_hist)
        b_hist = np.stack(b_hist)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path, a=a_hist, b=b_hist)
        return a_hist, b_hist
