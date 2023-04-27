# -*- coding: utf-8; -*-
"""Configuration for the Eulerian solid demo.

Simulation parameters, file paths, etcetera.

This file is configured for 316L steel.
"""

from enum import IntEnum
from fenics import Identity

# --------------------------------------------------------------------------------
# Physical parameters, common

# We model 316L steel as a Kelvin-Voigt viscoelastic material, with very small viscosity.
rho = 7581.62      # Density [kg/m³]
tau = 1e-5         # Kelvin-Voigt retardation time [s], defined by  η = tau E
# tau = 0          # You can set  τ = 0  to use a linear elastic model instead of Kelvin-Voigt.
ν = 0.33           # Poisson ratio [nondimensional]

# Velocity (constant) of co-moving frame in +x direction with respect to laboratory [m/s].
#
# To remain subcritical:  V0 < √(E / rho)    (longitudinal elastic wave speed)
#                         V0 < √(mu / rho)?  (shear wave speed)
# 316L steel: √(E / rho) = 4681.08952641054
#             mu = 6.38973e+10
#             √(mu / rho) = 2903.0884849832746
#
# V0 = 0.0  # Classical case (no axial motion), for debugging and comparison.
# V0 = 5e-2  # Typical L-PBF 3D printer laser focus spot velocity: 5 cm/s
V0 = 1.0
# V0 = 100.0
# V0 = 500.0  # near maximum stable for the thermomechanical variant

# V0 = 3300.0  # 3.3 km/s: maximum stable for steady state with fixed displacement (main01_solve)

# --------------------------------------------------------------------------------
# Physical parameters
#
# This section is for the pure mechanical solver, main01_solve.

E = 1.66133e+11                          # Young's modulus [Pa]
lamda = E * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu = E / (2 * (1 + ν))                   # shear modulus [Pa]

# --------------------------------------------------------------------------------
# Physical parameters
#
# This section is for the thermomechanical solver, main02_thermomech.

# ρ, ν, and τ are still constant in this model, but λ and μ may now depend on temperature.
# We also have the thermal parameters α, c, and k, which may depend on temperature.

# Plus we have some cooling-related parameters:
Γ = 26.2049          # Heat transfer coefficient [W/(m² K)], convective cooling for 316L steel in air, at v_air = 2 m/s
T_ext = 273.15 + 20  # Temperature of the environment [K]
H = 1e-3             # Thickness of the sheet [m]

# Output of 316L_with_cooling.py. The linear fits are very good, so we're
# unlikely to need anything more complicated for modeling these.
#
# The separate fit for η is unused - in our model,  η = τ E.
#
#     Model settings:
#     T₀ = 1700 K
#
#     For model with constant coefficients:
#     ρ [kg/m³] ≈ 7581.62
#     α [1/K] ≈ 2.08186e-05
#     c [J/(kg K)] ≈ 592.068
#     k [W/(m K)] ≈ 24.9607
#     E [Pa] ≈ 1.66133e+11
#     η [Pa s] ≈ 5.28819e+06
#
#     For model with linear coefficients:
#     ρ [kg/m³] ≈ 8078.28 - 0.49666 * T  [r = -0.996019, p = 4.88657e-15]
#     α [1/K] ≈ 1.75818e-05 + 3.08264e-09 * T  [r = 0.99926, p = 2.36902e-18]
#     c [J/(kg K)] ≈ 459.204 + 0.132864 * T  [r = 1, p = 1.96481e-129]
#     k [W/(m K)] ≈ 9.24638 + 0.0157143 * T  [r = 1, p = 2.7378e-45]
#     E [Pa] ≈ 2.2332e+11 - 8.18018e+07 * T  [r = -0.995759, p = 7.36947e-15]
#     η [Pa s] ≈ 7.10848e+06 - 2603.83 * T  [r = -0.995759, p = 7.36947e-15]

# We use the solidus temperature as the reference temperature.
T0 = 1700  # Reference temperature where thermal expansion is considered zero [K]

# Functions, T -> FEniCS `Expression`
E_func = lambda T: 2.2332e+11 - 8.18018e+07 * T                 # Young's modulus [Pa]
lamda_func = lambda T: E_func(T) * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu_func = lambda T: E_func(T) / (2 * (1 + ν))                   # shear modulus [Pa]

α_scalar = lambda T: 1.75818e-05 + 3.08264e-09 * T              # Thermal expansion coefficient, [1/K]
α_func = lambda T: Identity(2) * α_scalar(T)                    # Thermal expansion tensor, thermally isotropic material, 2D
dαdT_scalar = lambda T: 3.08264e-09                             # [1/K²]
dαdT_func = lambda T: Identity(2) * dαdT_scalar(T)

c_func = lambda T: 459.204 + 0.132864 * T                       # Specific heat capacity [J / (kg K)]
dcdT_func = lambda T: 0.132864                                  # [J / (kg K²)]

k_scalar = lambda T: 9.24638 + 0.0157143 * T                    # Heat conductivity coefficient [W / (m K)]
k_func = lambda T: Identity(2) * k_scalar(T)                    # Heat conductivity tensor, thermally isotropic material, 2D

# # DEBUG: functions, but outputting the constant values used in the constant-coefficient model.
# E_func = lambda T: 1.66133e+11                                  # Young's modulus [Pa]
# lamda_func = lambda T: E_func(T) * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
# mu_func = lambda T: E_func(T) / (2 * (1 + ν))                   # shear modulus [Pa]
#
# α_scalar = lambda T: 2.08186e-05                                # Thermal expansion coefficient, [1/K]
# α_func = lambda T: Identity(2) * α_scalar(T)                    # Thermal expansion tensor, thermally isotropic material, 2D
# dαdT_scalar = lambda T: 0.0                                     # [1/K²]
# dαdT_func = lambda T: Identity(2) * dαdT_scalar(T)
#
# c_func = lambda T: 592.068                                      # Specific heat capacity [J / (kg K)]
# dcdT_func = lambda T: 0.0                                       # [J / (kg K²)]
#
# k_scalar = lambda T: 24.9607                                   # Heat conductivity coefficient [W / (m K)]
# k_func = lambda T: Identity(2) * k_scalar(T)                    # Heat conductivity tensor, thermally isotropic material, 2D

# --------------------------------------------------------------------------------
# Numerical: common

# Enable streamline upwinding Petrov-Galerkin stabilization? (if the algorithm supports it)
enable_SUPG = True

# Draw element edges (and `extrafeathers` vis edges) when visualizing during simulation?
# show_mesh = True
show_mesh = False

# Solver mode (main01_solve only):
#   `True`: run dynamic simulation
#   `False`: solve for steady state
dynamic = True

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
# vis_every = 2.5 / 100
vis_every = 5.0 / 100

# Simulation end time
#
if V0 != 0.0:
    # Two transports of whole domain length. One to advect out the initial field that
    # might not be a solution of the steady-state PDE (so the results may not be reliable
    # until we get rid of it); one more to reach a steady state (with reasonable damping).
    T = 2 * (1 / V0)  # [s]
else:
    T = 2 * 0.01  # [s]

# Number of timesteps
#
# In main01_solve, the primal algorithm is expected to be A-stable, but for simulation quality,
# keep in mind the elastic Courant number:
#   Co = √(E / rho) * dt / he
# For example, for 316L steel, with T = 0.01, nt = 200, and N = 16 (per axis, quad grid), Co ≈ 3.7.
# Solve for `dt` to get `Co = 1`:
#   dt = he Co / √(E / rho) ≈ 1.33e-5  (at N = 16)
# so for T = 0.01 and N = 16,
#   nt = T / dt ≈ 750
#
# nt = 750

# # For main01_solve algorithms other than primal:
# nt = 6000  # for τ = 1e-5
# nt = 5000  # for τ = 1e-5, possible with step_adaptive (but slower due to adaptive algorithm)
# nt = 2000  # for τ = 0

# In main02_thermomech, the implicit midpoint rule (IMR, θ = 1/2) runs into stability issues.
# To stabilize the discrete equations, we use backward Euler time integration, which can use
# a rather large timestep. (Keep in mind the simulation quality, though.)
#
# The Courant number for the Eulerian thermal solver is:
#   Co = (V0 + du/dt) * dt / he ≈ V0 * dt / he
# The value is displayed during simulation by the main02_thermomech solver.
nt = 200

# Then the timestep size for the solver is:
dt = T / nt

# For main02_thermomech system iteration (weak coupling for multiphysics)
H1_tol = 1e-6  # Each field u must satisfy  ‖u - u_prev‖ < tol  separately. H1 norm.
maxit = 100    # Maximum number of system iterations in one timestep, before giving up.
# Usually, if everything is fine, the system converges in ≤ 15 iterations during the
# first few timesteps, and within ≤ 3 iterations as the simulation proceeds.

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

# The codes in this demo expect to be run from the top level of the project as e.g.
#   python -m demo.euleriansolid.main00_mesh
#   python -m demo.euleriansolid.main00_alternative_mesh
#   python -m demo.euleriansolid.main01_solve
#   python -m demo.euleriansolid.main02_thermomech
#   mpirun python -m demo.euleriansolid.main01_solve
#   mpirun python -m demo.euleriansolid.main02_thermomech
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/meshes/box.h5"  # for input and output

# For visualization in ParaView
vis_u_filename = "demo/output/euleriansolid/displacement.xdmf"
vis_v_filename = "demo/output/euleriansolid/velocity.xdmf"
vis_T_filename = "demo/output/euleriansolid/temperature.xdmf"
vis_dTdt_filename = "demo/output/euleriansolid/temperature_rate.xdmf"
vis_σ_filename = "demo/output/euleriansolid/stress.xdmf"
vis_vonMises_filename = "demo/output/euleriansolid/vonMises.xdmf"  # von Mises stress

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/euleriansolid/displacement_series"
sol_v_filename = "demo/output/euleriansolid/velocity_series"
sol_T_filename = "demo/output/euleriansolid/temperature_series"
sol_dTdt_filename = "demo/output/euleriansolid/temperature_rate_series"
sol_σ_filename = "demo/output/euleriansolid/stress_series"

# For auto-saving visualization screenshots (from dynamic solvers)
fig_output_dir = "demo/output/euleriansolid/"
fig_basename = "vis"
fig_format = "png"
