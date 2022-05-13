# -*- coding: utf-8; -*-
"""Configuration for the Eulerian solid demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical

# 316L steel
rho = 7581.62      # Density [kg/m³]
E = 1.66133e+11    # Young's modulus [Pa]
ν = 0.3            # Poisson ratio [nondimensional]  # TODO: check exact value for 316L
tau = 1e-5         # Kelvin-Voigt retardation time [s], defined by  η = tau E
# tau = 0          # Set tau to zero when instantiating solver to use the linear elastic model.

# # Debug test material (numerical scaling close to unity)
# rho = 1
# E = 10
# tau = 0.1

lamda = E * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu = E / (2 * (1 + ν))                   # shear modulus [Pa]

# Velocity of co-moving frame in +x direction (constant) [m/s]
#
# To remain subcritical:  V0 < √(E / rho)    (longitudinal elastic wave speed)
#                         V0 < √(mu / rho)?  (shear wave speed)
V0 = 5e-2  # Typical L-PBF 3D printer laser velocity
# V0 = 0.0  # Classical case (no axial motion), for debugging and comparison.

# --------------------------------------------------------------------------------
# Numerical

# For 316L steel
#
# Dynamic simulation end time [s]
T = 0.01
# Number of timesteps
nt = 6000  # for τ = 1e-5
# nt = 5000  # for τ = 1e-5, possible with step_adaptive
# nt = 2000  # for τ = 0

# For debug test material:
# T = 5.0
#   With T = 5.0, and a uniform mesh of 16×16 quads, V=Q1, Q=Q2;
#   then for linear elastic nt=1e3 works, but Kelvin-Voigt needs 1.25e4.
# nt = 12500
# nt = 1000

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

# For auto-saving visualization screenshots (from dynamic solver)
fig_output_dir = "demo/output/euleriansolid/"
fig_basename = "vis"
fig_format = "png"
