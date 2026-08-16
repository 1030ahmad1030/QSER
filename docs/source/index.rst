Welcome to QSER's documentation!
================================

QSER is a Data Physics tool for forward and inverse modeling and forecasting of physical systems.

**Authors**: Ahmad Muhammad and Fatih Külahcı

**Version**: 1.0.0

**License**: MIT

---

Quick Start
-----------

### Installation

.. code-block:: bash

   pip install QSER

### 1D Mesh and Gradient

.. code-block:: python

   import QSER as qs
   import numpy as np
   import matplotlib.pyplot as plt

   # Create mesh
   mesh = qs.Mesh.Structured1D(nx=100, L=10.0)
   x = mesh.get_cell_centers()

   # Create field
   field = np.sin(x)

   # Compute gradient
   from QSER.Operators import Gradient
   grad = Gradient(backend='numpy', method='5point')
   df_dx = grad.compute(field, dx=x[1]-x[0])

   # Plot
   plt.plot(x, field, label='sin(x)')
   plt.plot(x, df_dx, label='cos(x) (numerical)')
   plt.legend()
   plt.show()

### Energy Tracking

.. code-block:: python

   from QSER.Energy import EnergyTracker

   tracker = EnergyTracker(backend='numpy')
   eps_S = tracker.epsilon_S(S_field)
   eps_E = tracker.epsilon_E(E_field)
   eps_R = tracker.epsilon_R(R_field)

   print(f"Source Energy: {eps_S:.6f}")
   print(f"Environment Energy: {eps_E:.6f}")
   print(f"Response Energy: {eps_R:.6f}")

---

Core Concepts
-------------

The QSER framework is built on four fundamental equations:

.. math::

   R = S - E

.. math::

   L = L_c + L_d

.. math::

   L_c S = F

.. math::

   L E = L_d S

Where:

- **R** = Response (observed field)
- **S** = Source (conservative ghost field)
- **E** = Environment (memory accumulator)
- **Q** = Memory (Green's function)
- **L_c** = Conservative operator
- **L_d** = Dissipative operator

---

Features
--------

- **Mesh & Geometry** — Structured 1D/2D/3D meshes + STL/CAD geometry import
- **Operators** — Gradient, Laplacian, Divergence, Curl, TimeGradient
- **Backends** — NumPy, PyTorch, JAX, OpenFOAM
- **Energy Tracking** — ε_S, ε_E, ε_R, J_SE, D_E
- **QSER Decomposition** — R = S - E, L = Lc + Ld
- **Integration Engine** — Trapezoidal, Simpson, Monte Carlo, Quasi-Monte Carlo
- **Boundary Conditions** — Dirichlet, Neumann, Robin, Periodic, Wall, Source
- **Physics-Informed Neural Networks (PINNs)** — Mesh-free and mesh-based

---

Citation
--------

If you use QSER in your research, please cite:

.. code-block:: bibtex

   @software{muhammad2026qser,
     author = {Muhammad, Ahmad and K\"ulahc{\i}, Fatih},
     title = {QSER: A Data Physics Framework for Forward and Inverse Modeling of Physical Systems},
     year = {2026},
     url = {https://github.com/1030ahmad1030/QSER},
     version = {1.0.0},
     doi = {10.5281/zenodo.21965738}
   }

---

.. toctree::
   :maxdepth: 2
   :caption: Full Documentation (Coming Soon):

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
