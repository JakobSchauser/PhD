import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# Physical constants & setup
# ------------------------------------------------------------
N = 5                             # number of dipoles
G = 1.0                           # coupling constant (arbitrary units)
dtype = torch.float64

# Random initial positions and dipole moments
pos = torch.randn(N, 3, dtype=dtype, requires_grad=True)
dip = torch.randn(N, 3, dtype=dtype, requires_grad=True)


# ------------------------------------------------------------
# Dipole–dipole interaction energy
# ------------------------------------------------------------
def dipole_energy(positions, dipoles):
    # positions: (N,3)
    # dipoles: (N,3)
    diff = positions[:, None, :] - positions[None, :, :]   # (N,N,3)
    r = torch.linalg.norm(diff, dim=-1)                    # (N,N)
    r3 = r**3
    r5 = r**5

    # Avoid self-interaction singularities

    # dot products

    # dipole potential
    V = (
        mu_i_dot_mu_j / r3
        - 3 * (mu_i_dot_r * mu_j_dot_r) / r5
    )

    # Make a sum, only counting i<j terms (no double counting)


    return G * V_total


# Gradient check
E = dipole_energy(pos, dip)

# compute gradients (autograd)

print("Energy:", E.item())
print("Gradient on positions:\n", pos.grad)
print("Gradient on dipoles:\n", dip.grad)


# Simple time–stepping simulation (gradient descent style)
positions = pos.clone().detach().requires_grad_(True)

dipoles = dip.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([positions, dipoles], lr=0.01)

all_positions = []

for step in range(200):
    optimizer.zero_grad()
    energy = dipole_energy(positions, dipoles)
    energy.backward()
    optimizer.step()

    with torch.no_grad():
        dipoles /= torch.linalg.norm(dipoles, dim=-1, keepdim=True)

    all_positions.append(positions.detach().clone())

print("\nRelaxed positions:\n", positions.detach())
print("Relaxed dipoles:\n", dipoles.detach())

# ---------------