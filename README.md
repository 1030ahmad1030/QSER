

# QSER: Source-Environment-Response Framework

![QSER Logo](https://raw.githubusercontent.com/1030ahmad1030/QSER/main/https://raw.githubusercontent.com/1030ahmad1030/QSER/master/QSER/logo.png)

**A Data Physics Framework for Forward and Inverse Modeling of Physical Systems using Scientific Machine Learning, Classical and Hybrid Methods**

[![GitHub](https://img.shields.io/badge/GitHub-1030ahmad1030/QSER-blue)](https://github.com/1030ahmad1030/QSER)
[![PyPI](https://img.shields.io/badge/PyPI-QSER-blue)](https://pypi.org/project/QSER/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Website](https://img.shields.io/badge/Website-ahmadmuhammad325.com-blue)](https://ahmadmuhammad325.com/)

---

## About QSER

**The Memory "Q", Source "S", Environment "E", Response "R"**

**QSER** is a Data Physics tool for forward and inverse modeling and forecasting of physical systems. It includes QSER tools like:

- **Mesh & Geometry** — Structured 1D/2D/3D meshes + STL/CAD geometry import
- **Operators** — Gradient, Laplacian, Divergence, Curl, TimeGradient
- **Backends** — NumPy, PyTorch, JAX, OpenFOAM
- **Energy Tracking** — ε_S, ε_E, ε_R, J_SE, D_E
- **QSER Decomposition** — R = S - E, L = Lc + Ld
- **Integration Engine** — Trapezoidal, Simpson, Monte Carlo, Quasi-Monte Carlo
- **Boundary Conditions** — Dirichlet, Neumann, Robin, Periodic, Wall, Source
- **Physics Solver engines** 

---

## Core Equations

The QSER framework is built on four fundamental equations that lead to the extraction of system memory (Green's function "Q"):

| Equation | Description |
|----------|-------------|
| **R = S - E** | Fundamental decomposition of the observed field |
| **L = Lc + Ld** | Operator split (conservative + dissipative) |
| **Lc S = F** | Source equation — the conservative "ghost" field |
| **L E = Ld S** | Environment accumulator — the environment possesses the history of all interactions with the source and stores it in its memory "Q" |

### The Bank Account Analogy

Think of QSER like a bank account:

| Concept | QSER Equivalent | Meaning |
|---------|-----------------|---------|
| Salary | **S** (Source) | What you earn (conservative ghost) |
| Spending | **E** (Environment) | What you spend (memory accumulator) |
| Balance | **R** (Response) | What remains (observed field) |
| Bank Statement | **Q** (Memory) | Dictates the history of all transactions between the source and the environment |

The balance (Response) is always the difference between what you earn and what you spend:
**Balance = Salary — Spending** → **R = S - E**

---

## Installation

### Base Installation

```bash
pip install QSER
```

### With PyTorch Backend

```bash
pip install QSER[torch]
```

### With JAX Backend

```bash
pip install QSER[jax]
```

### With All Backends

```bash
pip install QSER[all]
```

### From Source

```bash
git clone https://github.com/1030ahmad1030/QSER.git
cd QSER
pip install -e .
```

---

## Quick Start

### 1D Mesh and Gradient

```python
import QSER as qs
import numpy as np
import matplotlib.pyplot as plt

# Create mesh
mesh = qs.Mesh.Structured1D(nx=100, L=10.0)
x = mesh.get_cell_centers()

# Create field
field = np.sin(x)

# Compute gradient (independent operator)
from QSER.Operators import Gradient
grad = Gradient(backend='numpy', method='5point')
df_dx = grad.compute(field, dx=x[1]-x[0])

# Plot
plt.plot(x, field, label='sin(x)')
plt.plot(x, df_dx, label='cos(x) (numerical)')
plt.legend()
plt.show()
```

### 2D Poisson Equation (Electrostatics)

```python
from QSER.Mesh import Structured2D
from QSER.Mesh.boundary import Boundary
from QSER.Operators import Laplacian
import numpy as np

# Create mesh
mesh = Structured2D(nx=50, ny=50, Lx=5.0, Ly=5.0)
X, Y = mesh.get_cell_centers()[:, :, 0], mesh.get_cell_centers()[:, :, 1]

# Charge density (Gaussian)
rho = np.exp(-((X-2.5)**2 + (Y-2.5)**2) / 0.5)

# Build Laplacian matrix
lap = Laplacian(mesh=mesh, backend='numpy', method='3point')
n = mesh.nx * mesh.ny
L = np.zeros((n, n))
for i in range(mesh.nx):
    for j in range(mesh.ny):
        idx = i * mesh.ny + j
        e = np.zeros((mesh.nx, mesh.ny))
        e[i, j] = 1.0
        L[:, idx] = lap.compute(e).flatten()

# Apply BCs and solve
# (Full example in tutorials)
```

### Energy Tracking

```python
from QSER.Energy import EnergyTracker

tracker = EnergyTracker(backend='numpy')
eps_S = tracker.epsilon_S(S_field)
eps_E = tracker.epsilon_E(E_field)
eps_R = tracker.epsilon_R(R_field)

print(f"Source Energy: {eps_S:.6f}")
print(f"Environment Energy: {eps_E:.6f}")
print(f"Response Energy: {eps_R:.6f}")
```

---

## Tutorials

| Notebook | Description |
|----------|-------------|
| `1D Mesh Tutorial` | Mesh creation, operators, boundary conditions |
| `2D Mesh and geometry` | 2D mesh, geometry, interpolation |
| `3D qser tutorial` | 3D mesh and operators |
| `Energy tutorial qser` | EnergyTracker, IntegrationEngine |
| `QSER tutorial 3` | PDE solvers (Poisson, Heat, Wave) |
| `QSER tutorial 4` | Advanced topics |

The notebooks are available in the `examples/` folder.

---

## Documentation

- **User Guide**: [https://qser.readthedocs.io](https://qser.readthedocs.io)
- **API Reference**: [https://qser.readthedocs.io/api](https://qser.readthedocs.io/api)
- **GitHub**: [https://github.com/1030ahmad1030/QSER](https://github.com/1030ahmad1030/QSER)
- **Website**: [https://www.ahmadmuhammad325.com](https://www.ahmadmuhammad325.com)

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If you use QSER in your research, please cite:

```bibtex
@software{muhammad2026qser,
  author = {Muhammad, Ahmad and K\"ulahc{\i}, Fatih},
  title = {QSER: A Data Physics Framework for Forward and Inverse Modeling of Physical Systems},
  year = {2026},
  url = {https://github.com/1030ahmad1030/QSER},
  version = {1.0.0}
}
```

---

## Acknowledgments

The authors acknowledge the use of **DeepSeek AI** as a development assistant in the implementation of this framework.

The authors thank **Qatar University** and **ASELSAN** for their support.

---

## Related Frameworks

- **QSignature**: Model-free dynamical regime classification for time series data ([GitHub](https://github.com/1030ahmad1030/QSignature))

---


