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

# tau = 0.1  # Kelvin-Voigt retardation time [s], defined as  tau := η / E
tau = 0  # set tau to zero when instantiating solver to use the linear elastic model

# To remain subcritical:  V0 < √(E / rho)    (longitudinal elastic wave speed)
#                         V0 < √(mu / rho)?  (shear wave speed)
#
# sqrt(10 / 1) ≈ 3.16
# sqrt((10 / (2 * 1.3)) / 1) ≈ 1.96
#
# It seems 1.75 is near the limit of this numerical scheme.
#
# TODO: The following is just speculation.
# Possible reason (linear elastic case): Courant number? Let Δt = 0.005 s.
#   N = 16  ->  L_elem = 1/16 m = 0.0625 m
# And since we use Q2 elements for the stress, the DOF spacing is actually
#   1/32 m = 0.03125 m
# On the other hand,
#   Eulerian wave speed in +x direction = V0 + sqrt(E / rho)
# Now if we set V0 = 1.75 m/s,
#   1/32 m / ((1.75 + 3.16) m/s) ≈ 0.00636 s
# is the time it takes a longitudinal wave to travel one DOF spacing in the +x direction.
#
V0 = 0.0           # velocity of co-moving frame in +x direction (constant) [m/s]

# --------------------------------------------------------------------------------
# Numerical

T = 5.0           # final time [s]

# With T = 5.0, and a uniform mesh of 16×16 quads, V=Q1, Q=Q2;
# then for linear elastic nt=1e3 works, but Kelvin-Voigt needs 2e4.
nt = 1000          # number of timesteps
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
