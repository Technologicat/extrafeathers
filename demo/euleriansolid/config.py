# -*- coding: utf-8; -*-
"""Configuration for the Eulerian solid demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical

# 316L steel
#rho = 7581.62      # density [kg/m³]
#E = 1.66133e+11    # Young's modulus [Pa]
rho = 1
E = 10
ν = 0.3            # Poisson ratio [nondimensional]  # TODO: check exact value for 316L

lamda = E * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu = E / (2 * (1 + ν))                   # shear modulus [Pa]

# To remain subcritical:  V0 < √(E / rho)  (longitudinal elastic wave speed)
V0 = 0.0           # velocity of co-moving frame in +x direction (constant) [m/s]

# --------------------------------------------------------------------------------
# Numerical

T = 5.0           # final time [s]

# On uniform mesh: at N=16, nt=500 works, but at N=32, nt=2500 is needed.
nt = 20000          # number of timesteps
dt = T / nt        # timestep size [s]

# --------------------------------------------------------------------------------
# Mesh

# These ID numbers must match the numbering in the .msh file (see the .geo file)
# so that the Gmsh-imported mesh works as expected.
class Boundaries(IntEnum):
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4
class Domains(IntEnum):
    STRUCTURE = 5

# --------------------------------------------------------------------------------
# File paths

# The solvers expect to be run from the top level of the project as e.g.
#   python -m demo.euleriansolid.main01_solve
# or
#   mpirun python -m demo.euleriansolid.main01_solve
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/meshes/box.h5"  # for input and output

# For visualization in ParaView
vis_u_filename = "demo/output/euleriansolid/displacement.xdmf"
vis_v_filename = "demo/output/euleriansolid/velocity.xdmf"
vis_σ_filename = "demo/output/euleriansolid/stress.xdmf"
vis_vonMises_filename = "demo/output/euleriansolid/vonMises.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/euleriansolid/displacement_series"
sol_v_filename = "demo/output/euleriansolid/velocity_series"
sol_σ_filename = "demo/output/euleriansolid/stress_series"
