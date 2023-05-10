# -*- coding: utf-8; -*-
"""Configuration for the Eulerian solid demo.

Simulation parameters, file paths, etcetera.

This file is configured for 316L steel.
"""

from enum import IntEnum
from fenics import Identity

# --------------------------------------------------------------------------------
# Geometry for internal mesh generator (main00_alternative_mesh.py)

# Length of domain [m], element density for mesh (equilateral triangles)
# L = 1.0
# elements_per_meter = 32
L = 2.0
elements_per_meter = 16

# Number of elements in the `x` direction. Note also the element degree, configured
# in the solver script itself. It is recommended to use P2 elements.
N = int(L * elements_per_meter)

# Aspect ratio (= width / height) of domain. `N` should be integer-divisible by this.
#
# TODO: Extreme aspect ratios (e.g. a few printed layers, over a length of many meters)
# are better handled by making a square domain, and adjusting the material parameters
# to make the simulation consider the material as orthotropic. We don't yet support an
# orthotropic stiffness tensor, so we should add that feature first.
aspect = 1

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
V0 = 5e-2  # Typical L-PBF 3D printer laser focus spot velocity: 5 cm/s
# V0 = 1.0
# V0 = 100.0
# V0 = 500.0  # near maximum stable for the thermomechanical variant

# V0 = 3300.0  # 3.3 km/s: maximum stable for steady state with fixed displacement (main01_solve)

# --------------------------------------------------------------------------------
# Physical parameters (basic solver)
#
# This section is for the pure mechanical solver, main01_solve.

E = 1.66133e+11                          # Young's modulus [Pa]
lamda = E * ν / ((1 + ν) * (1 - 2 * ν))  # first Lamé parameter [Pa]
mu = E / (2 * (1 + ν))                   # shear modulus [Pa]

# --------------------------------------------------------------------------------
# Physical parameters (advanced solver)
#
# This section is for the thermomechanical solver, main02_thermomech.

# The components of the multiphysics solver can be turned off individually.
# This is useful for:
#   - Experimenting with different boundary conditions for just one part of the problem
#     (to find good ones to use for the coupled problem).
#   - Numerical debugging, to see whether an issue arises from a subproblem, or from
#     the coupling between the subproblems.
#
# Disabling a component here will turn off the send of external fields into the disabled component,
# as well as the actual solve step for the disabled component (the field will remain at its initial value).
# The rest of the code will run the same, including any initialization and plotting.
#
# Except, this may affect what is plotted:
#   - When the thermal solver is enabled, the temperature and temperature material rate are plotted.
#   - When the thermal solver is disabled, these plots are replaced by elastic energy and kinetic energy
#     (eliminating empty subplot slots in the visualization sheet).
mechanical_solver_enabled = True
thermal_solver_enabled = True

# ρ, ν, and τ are still constant in this model, but λ and μ may now depend on temperature.
# We also have the thermal parameters α, c, and k, which may depend on temperature.

# Plus we have some cooling-related parameters:
# Γ = 26.2049  # Heat transfer coefficient [W/(m² K)], convective cooling for 316L steel in air, at v_air = 2 m/s
# T_ext = 273.15 + 20  # Temperature of the environment [K]
# H = 1e-3             # Thickness of the sheet [m]
#
# Values consistent with our 1D study:
# Γ = 10.0  # [W/(m² K)]  # single-sided cooling: one side exposed
Γ = 2 * 10.0   # [W/(m² K)]  # double-sided cooling: both sides exposed (absorb the 2 here)
T_ext = 273.15 + 22  # [K]
H = 50e-6  # [m]

# Output of 316L_with_cooling.py. The linear fits are very good, so we're
# unlikely to need anything more complicated for modeling these.
#
# The separate fit for η is unused - in our model,  η = τ E.
# Also, in our model, ρ is constant.
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
# Parameters for inlet temperature profile simulation (advanced solver).
#
# When the advanced solver starts, it invokes an oversimplified 0D cooling simulation, with the same `ρ` and `c`,
# to estimate the inlet temperature profile as a function of depth. See `initial_T_profile.py`.
#
# The parameter `inlet_profile_scale` controls the mapping of the 0D simulation time coordinate to the depth coordinate.
# Considering the individual printed layers, and making a continuous approximation, we have:
#
#   layer_number = t / laser_return_time
#   depth = layer_number * layer_thickness
#
# where `t` is the 0D cooling simulation time coordinate. Combining these,
#
#   depth = t * layer_thickness / laser_return_time
#
# `layer_thickness`: grain size is ~50μm in diameter. That's approximately also the layer thickness,
# if (for the purposes of constructing this mapping) we neglect thermal shrinkage and the removal of pores.
#
# `laser_return_time` is how long until the laser sweeps the same spot again.
#
# Let us define the *relative* 0D simulation time as
#
#   relt = t / inlet_profile_tmax
#
# so that the simulation end time is at `relt = 1`. Inserting this definition, we have
#
#   depth = (relt * inlet_profile_tmax) * layer_thickness / laser_return_time
#
# Dividing both sides by the domain height:
#
#   reldepth = relt * (inlet_profile_tmax / domain_height) * layer_thickness / laser_return_time
#
# Reorganizing,
#
#   reldepth = relt * (layer_thickness / domain_height) * (inlet_profile_tmax / laser_return_time)
#
# from which we extract the final definition for the nondimensional scaling parameter of the inlet profile:
#
#   relt = reldepth * inlet_profile_scale,
#   inlet_profile_scale = (laser_return_time / inlet_profile_tmax) / (layer_thickness / domain_height)
#
# Note that because we create the inlet profile by interpolating the 0D simulation output, no data is available for `relt > 1`.
# Thus we must require
#
#   relt ∈ [0, 1]
#
# On the other hand, by definition of the relative depth,
#
#   reldepth ∈ [0, 1]
#
# Also, as was seen, `relt` is linearly proportional to `reldepth`. Therefore, to satisfy both ranges, it must hold that
#
#   inlet_profile_scale ≤ 1
#
# or in other words,
#
#   (laser_return_time / inlet_profile_tmax) / (layer_thickness / domain_height) ≤ 1
#
# Rearranging, a 0D simulation of length `inlet_profile_tmax` can support laser return times up to:
#
#   laser_return_time ≤ (layer_thickness / domain_height) * inlet_profile_tmax
#
# or alternatively, the required 0D simulation length for given `laser_return_time` is:
#
#   inlet_profile_tmax ≥ laser_return_time * (domain_height / layer_thickness)
#
# The optimal scaling is at the equality - i.e. choosing `inlet_profile_tmax` (how long the 0D simulation
# needs to be) to make `inlet_profile_scale = 1`, so as to use the complete 0D simulation output as the
# inlet temperature profile.
#
# Thus, we fix `inlet_profile_scale = 1` without making it into a parameter.
#
# What is the value we should choose for the laser return time? At H = 50 µm, the printable area depth
# 10 cm = 2000 layers. The simulated domain (to approximate a semi-infinite sheet, to look at the process
# in the absence of any boundary effects), which is 2 m in depth, is 40k layers.
#
# The simulated 0D cooling process is so fast that the metal reaches room temperature within 20 s,
# which is obviously overestimating the cooling rate.
#
# So to actually have anything other than `T_ext` as the temperature of the nodes other than the one
# in the upper left corner, we need an unrealistically fast laser return time (considering that in
# an L-PBF machine, even if we are printing this one thin fin only, recoating takes several seconds),
# so that the entire domain depth represents not much more than 20 s. Maybe 200 s is ok (so cooling
# to room temperature, at the domain inlet, happens across the first 10% of the domain depth),
# but 2000 s (cooling across the first 1% of domain depth) definitely isn't.
#
# So as a compromise, let's go for `inlet_profile_tmax = 200` s. At H = 50 µm and domain_height = 2 m,
# this means 40k layers in 200 s, so the laser return time is 200 s / 40e3 = 5 ms - which is three
# orders of magnitude too short, so there's definitely room for improvement here. (Maybe use a graded mesh?)
inlet_profile_tmax = 200.0  # [s]

# # Alternatively, we could set this up like in the above comments:
# # How long until the laser sweeps the same spot again (but on the next layer).
# laser_return_time = 1.0  # [s]
#
# # Geometry related to the inlet temperature profile.
# domain_height = L / aspect  # [m]   TODO/NOTE: only works for internal mesh generator
# layer_thickness = H  # [m]
#
# # End time of the 0D cooling simulation. [s]
# inlet_profile_tmax = laser_return_time * (domain_height / layer_thickness)

# Radius of a representative, spherical powder grain.
R = H / 2  # [m]

# --------------------------------------------------------------------------------
# Numerical: common

# Enable streamline upwinding Petrov-Galerkin stabilization? (if the algorithm supports it)
enable_SUPG = True

# Draw element edges (and `extrafeathers` vis edges) when visualizing during simulation?
# show_mesh = True
show_mesh = False

# Project strain, stress, elastic energy onto the main function space (`V`) for online (during simulation )visualization?
# Exported fields are not affected.
#
# This is useful when `V` is P1, which naturally makes those fields dG0. Projecting a dG0 field onto a P1 space effectively
# patch-averages it, thus reducing checkerboard artifacts.
project_lower_degree_fields_to_V = False

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
    #
    # Note this `L` is the domain length for the internal mesh generator; if using
    # an external mesh, the value should be calculated as `xmax - xmin` manually.
    T = 2 * (L / V0)  # [s]
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
maxit = 20    # Maximum number of system iterations in one timestep, before giving up.
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
vis_εu_filename = "demo/output/euleriansolid/strain.xdmf"
vis_εv_filename = "demo/output/euleriansolid/strain_rate.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/euleriansolid/displacement_series"
sol_v_filename = "demo/output/euleriansolid/velocity_series"
sol_T_filename = "demo/output/euleriansolid/temperature_series"
sol_dTdt_filename = "demo/output/euleriansolid/temperature_rate_series"
sol_σ_filename = "demo/output/euleriansolid/stress_series"
sol_εu_filename = "demo/output/euleriansolid/strain_series"
sol_εv_filename = "demo/output/euleriansolid/strain_rate_series"

# For auto-saving visualization screenshots (from dynamic solvers)
fig_output_dir = "demo/output/euleriansolid/"
fig_basename = "vis"
fig_format = "png"
