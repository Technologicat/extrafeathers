# -*- coding: utf-8; -*-
"""Configuration for the Eulerian solid demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical parameters

# 316L steel
rho = 7581.62      # Density [kg/m³]
E = 1.66133e+11    # Young's modulus [Pa]
ν = 0.3            # Poisson ratio [nondimensional]  # TODO: check exact value for 316L
tau = 1e-5         # Kelvin-Voigt retardation time [s], defined by  η = tau E
# tau = 0          # Set tau to zero when instantiating solver to use the linear elastic model.
# tau = 1.0  # DEBUG testing

# # Debug test material (numerical scaling close to unity)
# rho = 1
# E = 10
# tau = 0.1

# Velocity of co-moving frame in +x direction (constant) [m/s]
#
# To remain subcritical:  V0 < √(E / rho)    (longitudinal elastic wave speed)
#                         V0 < √(mu / rho)?  (shear wave speed)
# 316L steel: √(E / rho) = 4681.08952641054
#             mu = 6.38973e+10
#             √(mu / rho) = 2903.0884849832746
V0 = 5e-2  # Typical L-PBF 3D printer laser velocity: 5 cm/s
# V0 = 3300.0  # 3.3 km/s: maximum stable for 316L steel, for steady state with fixed displacement
# V0 = 0.0  # Classical case (no axial motion), for debugging and comparison.

# Then the Lamé parameters for the solver are:
lamda = E * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu = E / (2 * (1 + ν))                   # shear modulus [Pa]

# --------------------------------------------------------------------------------
# Numerical: general

# Solver mode:
#   `True`: run dynamic simulation
#   `False`: solve for steady state
dynamic = True

# Enable streamline upwinding Petrov-Galerkin stabilization? (if the algorithm supports it)
enable_SUPG = True

# Draw element edges (and `extrafeathers` vis edges) when visualizing during simulation?
show_mesh = True

# --------------------------------------------------------------------------------
# Numerical: dynamic simulation

# How many timesteps, at most, to export to `.xdmf` from the whole simulation.
# If `nsave_total ≥ nt`, all timesteps are exported.
nsave_total = 1000

# How often to update the online (during simulation) visualization, as a ratio in [0, 1].
# This visualization is meant to provide visual feedback at a reasonable latency on the
# behavior of the simulation, without the need to load the data into ParaView to look at
# the `.xdmf`.
#
# Each visualization is auto-saved as an image (see `fig_*` settings further below),
# to preserve everything that was plotted for future visual inspection.
#
# Plotting is slow, so for fast simulation, the visualization should be updated
# rather rarely. However, since we have a vibrating system, visualizing too rarely
# may miss important temporal detail in the sequence of screenshots produced.
#
# For example, if the visualization interval happens to match an eigenfrequency,
# in the sequence of screenshots it may look like nothing is happening, whereas
# in reality, the object vibrates for an integer number cycles between each two
# successive screenshots.
#
# Note that this setting does NOT affect the data export to xdmf; this controls
# only the plotting behavior.
vis_every = 2.5 / 100

# For 316L steel
#
# Simulation end time [s]
T = 0.01

# Number of timesteps
#
# The primal algorithm is expected to be A-stable, but for simulation quality,
# keep in mind the elastic Courant number:
#   Co = √(E / rho) * dt / he
# For example, for 316L steel, with T = 0.01, nt = 200, and N = 16 (per axis, quad grid), Co ≈ 3.7.
# Solve for `dt` to get `Co = 1`:
#   dt = he Co / √(E / rho) ≈ 1.33e-5  (at N = 16)
# so for T = 0.01 and N = 16,
#   nt = T / dt ≈ 750
#
nt = 750

# # For algorithms other than primal:
# nt = 6000  # for τ = 1e-5
# nt = 5000  # for τ = 1e-5, possible with step_adaptive (but slower due to adaptive algorithm)
# nt = 2000  # for τ = 0

# # For debug test material:
# T = 5.0
# # With T = 5.0, and a uniform mesh of 16×16 quads, V=Q1, Q=Q2;
# # then for linear elastic nt=1e3 works, but Kelvin-Voigt needs 1.25e4.
# nt = 12500
# nt = 1000

# Then the timestep size for the solver is:
dt = T / nt

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
