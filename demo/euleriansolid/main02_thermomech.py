# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of axial motion).

Thermomechanical variant; models also thermal expansion.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer, Popper

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
                    DirichletBC,
                    Constant, Function,
                    project, FunctionAssigner,
                    Vector,
                    tr, Identity, sqrt, inner, dot,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    norm,
                    begin, end,
                    parameters)

# custom utilities for FEniCS
from extrafeathers import common
from extrafeathers import meshfunction
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import (LinearMomentumBalance,
                                InternalEnergyBalance)
from extrafeathers.pdes.eulerian_solid_advanced import ε
from extrafeathers.pdes.numutil import mag, Minn
from .config import (rho, tau, V0, T0, Γ, T_ext, H, dt, nt, T, H1_tol, maxit,
                     E_func, lamda_func, mu_func, α_func, dαdT_func, c_func, dcdT_func, k_func,
                     nsave_total, vis_every, enable_SUPG, show_mesh, project_lower_degree_fields_to_V,
                     mechanical_solver_enabled, thermal_solver_enabled,
                     Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_v_filename, sol_v_filename,
                     vis_T_filename, sol_T_filename,
                     vis_dTdt_filename, sol_dTdt_filename,
                     vis_σ_filename, sol_σ_filename,
                     vis_εu_filename, sol_εu_filename,
                     vis_εv_filename, sol_εv_filename,
                     vis_vonMises_filename,
                     fig_output_dir, fig_basename, fig_format)


my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces.
#
# V is our primary space. We use linear elements for simplicity and speed.
# Q is an auxiliary space, used for visualization of some of the results.
#
# Note  ε ∝ ∇u,  so if  u ∈ H¹ ⊂ C⁰,  then naturally  ε ∈ C⁻¹.
# Similarly for dε/dt and du/dt. And since (Kelvin-Voigt material)
#     σ = K : ε + τ K : dε/dt
# it follows that also  σ ∈ C⁻¹.
#
# Thus when V is P1, the natural choice for Q is the piecewise constant space DP0.
# When V is P2, the natural choice for Q is P1.
#
# Also note that this choice would not satisfy the relevant inf-sup condition
# were we to use a mixed method to solve the linear momentum balance (in
# classical structural mechanics, the space for Cauchy stress needs to have
# many more DOFs than the space for displacement). But we actually solve using
# a primal formulation, and Q is used only for visualizing results, so it's ok.
#
# The solvers use a mixed function space, built from copies of these spaces:
#   - `u` is allocated one `V_rank1` space, and `du/dt` is allocated another.
#   - `T` is allocated one `V_rank0` space, and `dT/dt` is allocated another.
#
# The rank-0 spaces are also used in visualization, when there is a need to L2-project
# a scalar expression into a FEM space suitable for plotting.
#
# Using P2 elements for V is recommended; P1 tends to produce bad gradients (strain!)
# with checkerboard oscillations.
#
# Note that P2 fields are exported as the corresponding full nodal resolution P1 field.
# The strain visualization computed by ParaView (from the displacement, using the
# Compute Derivatives filter) is still dP0, and looks bad (checkerboard).
# But the P1 stress/strain fields exported by this solver are fine.
V_rank1 = VectorFunctionSpace(mesh, 'P', 2)
V_rank0 = V_rank1.sub(0).collapse()
Q_rank2 = TensorFunctionSpace(mesh, 'P', 1)
# Q_rank2 = TensorFunctionSpace(mesh, 'DP', 0)  # Use this when V is P1.
Q_rank0 = Q_rank2.sub(0).collapse()

# Function space of ℝ (single global DOF). By projecting onto this space,
# we can compute summary statistics over the whole domain.
W = FunctionSpace(mesh, "R", 0)

# Start by detecting the bounding box - this can be used e.g. for fixing the
# displacement on a line inside the domain.
with timer() as tim:
    ignored_cells, nodes_dict = meshmagic.all_cells(V_rank0)
    ignored_dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])

    domain_length = xmax - xmin
    domain_width = ymax - ymin
    A = domain_length * domain_width
    v_el_T0 = (E_func(T0) / rho)**0.5

    he = meshfunction.cell_mf_to_expression(meshfunction.meshsize(mesh))
    mean_he = float(project(he, W)) / A

if my_rank == 0:
    print(f"Geometry detection completed in {tim.dt:0.6g} seconds.")
    print(f"x ∈ [{xmin:0.6g}, {xmax:0.6g}] m, y ∈ [{ymin:0.6g}, {ymax:0.6g}] m.")
    print(f"Domain length L = {domain_length:0.6g} m; simulation end time T = {T} s; timestep Δt = {dt} s")

    print(f"Mean of longest edge length in mesh {mean_he:0.6g} m")
    print(f"With elements of mean size: {1 / mean_he:0.6g} el/m; {domain_length / mean_he:0.6g} el over domain length")

    print(f"At reference temperature T0 = {T0:0.6g} K, Young modulus E(T0) = {E_func(T0):0.6g} Pa")
    print("Longitudinal elastic waves:")
    print(f"    Propagation speed at reference temperature v_el(T0) = {v_el_T0:0.6g} m/s (one domain length in {domain_length / v_el_T0:0.6g} s)")
    print(f"    Courant number at reference temperature Co(T0) = {v_el_T0 * dt / mean_he:0.6g}")

# --------------------------------------------------------------------------------
# Set up the solvers

# Boundary condition lists.
#
# We create the lists now, because the solver constructor needs to store a reference to the list instance.
# But to actually set up Dirichlet BCs, we need a reference to the subspaces created by solver initialization.
# So we will fill these lists later. For consistency of presentation, we fill in also the Neumann BCs later.

# Dirichlet BCs
bcu = []  # displacement [m]
bcT = []  # temperature [K]

# Neumann BCs
bcσ = []  # Cauchy stress tensor [Pa], projected automatically to the direction of the outer normal
bcq = []  # scalar heat flux in direction of outer normal [W/m²]

# Instantiate the solvers for our multiphysics problem.
# We use backward Euler time integration (θ = 1) to help stabilize the numerics.
linmom_solver = LinearMomentumBalance(V_rank1, Q_rank2, Q_rank2,
                                      rho, lamda_func, mu_func, tau,
                                      α_func, dαdT_func, T0,
                                      V0,
                                      bcu, bcσ, dt, θ=1.0,
                                      boundary_parts=boundary_parts)
thermal_solver = InternalEnergyBalance(V_rank0,
                                       rho, c_func, dcdT_func, k_func, T0,
                                       bcT, bcq, dt, θ=1.0,
                                       advection="general", use_stress=True,
                                       boundary_parts=boundary_parts)

# Plotting labels for the rate operator. This model uses the advective rate "d/dt".
#   - The displacement solver is mixed Lagrangean-Eulerian (MLE); the advection velocity is the axial drive velocity.
#     The displacement `u` and the material parcel velocity `du/dt` are measured against the axially co-moving frame.
#   - The thermal solver is pure Eulerian; the advection velocity is the full velocity of the material parcels in the
#     laboratory frame. The temperature rate `dT/dt` is the *material* derivative of the temperature `T`.
dlatex = r"\mathrm{d}"
dtext = "d"

if my_rank == 0:
    print(f"Number of DOFs: u {V_rank1.dim()}, {dtext}u/{dtext}t {V_rank1.dim()}, T {V_rank0.dim()}, {dtext}T/{dtext}t {V_rank0.dim()}")

# NOTE: Accessing the `.sub(j)` of a mixed field (e.g. `s_` here) seems to create a new copy of the subfield every time.
# (Indeed, look at `dolfin.function.Function.sub` and `dolfin.function.Function.__init__`.)
# So e.g. `fields["T"]` gives read access; but to write to a subfield of a mixed field, we must use a `FunctionAssigner`.
# Examples further below.
fields = {"u": linmom_solver.s_.sub(0),
          "du/dt": linmom_solver.s_.sub(1),
          "T": thermal_solver.s_.sub(0),
          "dT/dt": thermal_solver.s_.sub(1),
          "σ": linmom_solver.σ_,
          "ε": linmom_solver.εu_,
          "dε/dt": linmom_solver.εv_}
subspaces = {k: v.function_space() for k, v in fields.items()}  # for setting boundary conditions

# --------------------------------------------------------------------------------
# Boundary conditions, mechanical subproblem

σ0 = 1e8  # Axial pull strength, [Pa], for examples that use it.

# These are used only by cases with a time-dependent boundary condition,
# but the main loop expects the variables to exist (to detect whether to use them or not).
u0_left = None
u0_right = None
u0_func = lambda t: 0.0

σ0_left = None
σ0_right = None
σ0_func = lambda t: 0.0

# Here are some examples of how to make a time-dependent boundary condition for `u`.
# The value can then be updated by e.g. `u0_right.u0 = u0_func(t)`.
#
# # Left and right edges: fixed left end, displacement-controlled pull at right end
# bcu_left = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
# from fenics import Expression
# u0_func = lambda t: 1e-2 * t
# u0_right = Expression(("u0", "0"), degree=1, u0=u0_func(0.0))
# bcu_right = DirichletBC(subspaces["u"], u0_right, boundary_parts, Boundaries.RIGHT.value)
# bcu.append(bcu_left)
# bcu.append(bcu_right)

# # Left and right edges: fixed left end, displacement-controlled *u1 only* at right end.
# # Since we don't set the u2 component by a Dirichlet BC, it gets the Neumann BC for stress.
# bcu_left = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
# from fenics import Expression
# u0_func = lambda t: 1e-2 * t
# u0_right = Expression("u0", degree=1, u0=u0_func(0.0))
# bcu_right = DirichletBC(subspaces["u"].sub(0), u0_right, boundary_parts, Boundaries.RIGHT.value)  # u1
# bcu.append(bcu_left)
# bcu.append(bcu_right)

# # Left and right edges: displacement-controlled pull
# # `dolfin.Expression` compiles to C++, so we must define these separately. Trying to flip the sign
# # of an `Expression` and setting that to a `DirichletBC` causes a one-time `project` to take place.
# # That won't even work here, but even if it did, it wouldn't give us an updatable.
# from fenics import Expression
# u0_func = lambda t: 1e-2 * t
# u0_left = Expression(("-u0", "0"), degree=1, u0=u0_func(0.0))
# u0_right = Expression(("+u0", "0"), degree=1, u0=u0_func(0.0))
# bcu_left = DirichletBC(subspaces["u"], u0_left, boundary_parts, Boundaries.LEFT.value)
# bcu_right = DirichletBC(subspaces["u"], u0_right, boundary_parts, Boundaries.RIGHT.value)
# bcu.append(bcu_left)
# bcu.append(bcu_right)

# # Fixed left end (with stress-controlled pull at right end, matches Kurki et al., 2016).
# bcu_left = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
# bcu.append(bcu_left)

# # Fixed bottom edge (3D printing)
# bcu_bottom = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.BOTTOM.value)
# bcu.append(bcu_bottom)

# 3D printing: u1 fixed at left edge, u2 fixed at bottom edge
# bcu1_left = DirichletBC(subspaces["u"].sub(0), Constant(0), boundary_parts, Boundaries.LEFT.value)
# bcu.append(bcu1_left)
bcu2_bottom = DirichletBC(subspaces["u"].sub(1), Constant(0), boundary_parts, Boundaries.BOTTOM.value)
bcu.append(bcu2_bottom)

# 3D printing: u fixed at upper left corner (where the material exits the laser focus spot and has just solidified)
from fenics import CompiledSubDomain
upper_left = CompiledSubDomain(f"near(x[0], {xmin}) && near(x[1], {ymax})")
bcu1_upperleft = DirichletBC(subspaces["u"].sub(0), Constant(0), upper_left, method="pointwise")
bcu.append(bcu1_upperleft)
# bcu2_upperleft = DirichletBC(subspaces["u"].sub(1), Constant(0), upper_left, method="pointwise")
# bcu.append(bcu2_upperleft)

# The stress [Pa] uses a Neumann BC, with the boundary stress field set here.
# The Cauchy stress tensor given here is evaluated (and automatically projected
# into the outer normal direction) on the boundaries that have no Dirichlet boundary
# condition on `u`.
#
# The format for Neumann BCs in the advanced solver is [(fenics_expression, boundary_tag or None), ...].
# The boundary tags are as in `boundary_parts`, and `None` means "apply this BC to the whole Neumann boundary".

# # Heaviside step load at right edge at t = 0
# bcσ.append((Constant(((σ0, 0), (0, 0))), Boundaries.RIGHT.value))

# # Ramp-up: linearly increase the load to its full value during the first 10% of the simulation, then stay at the full value.
# from fenics import Expression
# σ0_func = lambda t: min(10 * (t / T), 1.0) * σ0
# # σ0_left = Expression((("σ0", "0"), ("0", "0")), σ0=σ0_func(0.0), degree=1)
# # bcσ.append((σ0_left, Boundaries.LEFT.value))
# σ0_right = Expression((("σ0", "0"), ("0", "0")), σ0=σ0_func(0.0), degree=1)
# bcσ.append((σ0_right, Boundaries.RIGHT.value))

# # Pure Neumann BCs for the mechanical problem don't work as-is, we would need a rigid-body mode remover
# # to prevent a runaway excitation. But obviously, if we fix the displacement at one point, the solution
# # will be unique. Note that to do this in the interior of the domain needs a regular mesh to work
# # properly, because we need to have a DOF exactly (up to rounding) at the point where we are fixing the
# # displacement.
# #
# # How to apply a "pointwise BC":
# #   https://fenicsproject.org/qa/10273/pointwise-bc/
# #
# # This is missing from the latest docs; see old docs.
# #   https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/fem/bcs/DirichletBC.html
# #   https://fenicsproject.org/olddocs/dolfin/latest/python/_autogenerated/dolfin.cpp.fem.html#dolfin.cpp.fem.DirichletBC
# #
# # From the 1.3.0 docs linked above; quoted here for preservation:
# #     The ‘method’ variable may be used to specify the type of method used to identify degrees of freedom
# #     on the boundary. Available methods are: topological approach (default), geometric approach, and
# #     pointwise approach. The topological approach is faster, but will only identify degrees of freedom
# #     that are located on a facet that is entirely on the boundary. In particular, the topological
# #     approach will not identify degrees of freedom for discontinuous elements (which are all internal to
# #     the cell). A remedy for this is to use the geometric approach. In the geometric approach, each dof
# #     on each facet that matches the boundary condition will be checked. To apply pointwise boundary
# #     conditions e.g. pointloads, one will have to use the pointwise approach which in turn is the
# #     slowest of the three possible methods. The three possibilties are “topological”, “geometric” and
# #     “pointwise”.
# from fenics import CompiledSubDomain
# xmid = (xmin + xmax) / 2
# ymid = (ymin + ymax) / 2
# # center_vline = CompiledSubDomain(f"near(x[0], {xmid})")
# # center_hline = CompiledSubDomain(f"near(x[1], {ymid})")
# center_point = CompiledSubDomain(f"near(x[0], {xmid}) && near(x[1], {ymid})")
# # center_left = CompiledSubDomain(f"near(x[0], {xmin}) && near(x[1], {ymid})")
# bcu_center1 = DirichletBC(subspaces["u"].sub(0), Constant(0), center_point, method="pointwise")  # u1(0, 0) = 0
# bcu_center2 = DirichletBC(subspaces["u"].sub(1), Constant(0), center_point, method="pointwise")  # u2(0, 0) = 0
# bcu.append(bcu_center1)
# bcu.append(bcu_center2)

# Recompile to refresh the Neumann BCs
linmom_solver.compile_forms()

# --------------------------------------------------------------------------------
# Boundary conditions, thermal subproblem

T_left = T0
# T_right = T0
T_bottom = T0 - 100.0

# Axially moving continuum: specify the temperature of the material parcels that enter the domain at the left.
# Don't set anything at the right - the default zero Neumann (no change in temperature in axial direction i.e. steady outflow) is appropriate.
#
# Note the same `T_profile` is used also for setting the initial condition for the temperature field.

# # Constant temperature (wrong, but simple, for testing/debugging)
# T_profile = project(Constant(T_left), V_rank0)

# # Linear temperature profile (wrong, but simple, for testing/debugging)
# from fenics import Expression
# T_profile = Expression("T_bottom + (T_left - T_bottom) * (x[1] + 0.5)", degree=1, T_bottom=T_bottom, T_left=T_left)

# A somewhat more advanced approach:
#
# Roughly estimate the inlet temperature profile by performing an oversimplified 0D cooling simulation,
# with the same material parameters as for the main simulation. The solution of that 0D simulation
# specifies the inlet temperature profile.
#
# Obviously, we can use any input data here, so we can improve this by improving the simplified simulation.
#
# Can be implemented via a `UserExpression` - slow, but works. Speed doesn't matter much here.
# The inlet profile is only evaluated for the left edge (for the boundary condition), and once
# in the whole domain (for the initial condition).
#
# How to:
#   https://fenicsproject.discourse.group/t/interpolate-numpy-ndarray-to-function/6167/2
#
from scipy.interpolate import interp1d
from fenics import UserExpression
from . import initial_T_profile
profile_tmax = 20.0  # [s], end time of the 0D cooling simulation
_, tt, TT = initial_T_profile.estimate(tmax=profile_tmax)  # <-- the important part
T_inlet = interp1d(tt, TT, fill_value=(TT[0], TT[-1]))
class InletTemperatureProfile(UserExpression):
    def __init__(self, degree=2, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        # extract the y coordinate
        if x.shape == (2,):
            y = x[1]
        else:
            y = x

        rely = (y - ymin) / (ymax - ymin)
        relt = 1.0 - rely  # hot at the top surface (ymax <-> tmin)
        # TODO: Scale the cooling time coordinate sensibly. Consider how long until the laser sweeps again.
        # Grain size is ~50μm in diameter, so that's approximately also the thickness of one layer (neglecting thermal shrinkage, and the removal of pores).
        # dtdrelt = profile_tmax  # full time range (i.e. y at bottom maps to simulation end time in 0D cooling simulation)
        dtdrelt = 0.5 * profile_tmax  # half of the time range (i.e. y at bottom maps to halfway to simulation end time in 0D cooling simulation)
        t = dtdrelt * relt
        values[0] = T_inlet(t)

    def value_shape(self):
        return ()
T_profile = InletTemperatureProfile()


# Whichever temperature profile we defined above, apply it at the inlet boundary as a Dirichlet BC:
bcT_left = DirichletBC(subspaces["T"], T_profile, boundary_parts, Boundaries.LEFT.value)
bcT.append(bcT_left)


# The heat flux [W/m²] uses uses a Neumann BC, with the boundary scalar flux
# (in the direction of the outer normal) set here. Same format as in the mechanical solver.
#
# # To use a zero Neumann BC, which is the classic do-nothing BC, we can simply omit the term.
# # The solver will then run marginally faster, as the equation doesn't include this term.
# bcq.append((Constant(0), None))

# # Cooling at upper edge. Generic FEM function, to be refreshed with data in `update_cooling`.
# q_upper = Function(V_rank0)
# bcq.append((q_upper, Boundaries.TOP.value))

# Another way to do this: since this boundary condition is linear in `T`, we can use `thermal_solver.u`
# to insert the Galerkin series of `T`. Then it'll automatically use the latest data.
#
# Note the extra factor of `H`, because this is a 2D model: [Γ] = W/m², and the boundary integration
# only eliminates one `m`.
bcq.append((Constant(-Γ) * (thermal_solver.u - Constant(T_ext)), Boundaries.TOP.value))  # [W/m²]

# # Higher powers (Stefan-Boltzmann radiative cooling) can be done similarly.
# # Split off one `T` to use its Galerkin series, and provide the rest as data:
# T3 = Function(V_rank0)
# bcq.append((Constant(-Γ) * (thermal_solver.u * T3 - Constant(T_ext)**4), Boundaries.TOP.value))  # [W/m²]
# # ...and then update `T3` in `update_cooling`, as `T3.assign(project(fields["T"]**3, V_rank0))`

# Recompile to refresh the Neumann BCs
thermal_solver.compile_forms()

# --------------------------------------------------------------------------------
# Initial conditions, mechanical subproblem

# In our examples the initial field for `u` is zero, which is also the default.

# --------------------------------------------------------------------------------
# Initial conditions, thermal subproblem

# A linear function of x is at least a trivial steady-state solution of the standard heat equation,
# so we could use something like that.

# # In early versions, our domain used to be Ω = (-0.5, 0.5)².
# from fenics import Expression
# initial_T = project(Expression("T_left + (T_right - T_left) * (x[0] + 0.5)", degree=1, T_left=T_left, T_right=T_right), V_rank0)

# But in the axially moving case, we don't actually know how much the material
# naturally cools in a steady state as it travels through the domain - whereas
# the dynamic simulation will attempt to reach that steady state. So it is better
# to initialize to a uniform temperature field, which will likely get us there faster.
#
# But even better in that regard is to use the inlet temperature profile.
initial_T = project(T_profile, V_rank0)
initial_dTdt = Function(V_rank0)  # zeroes

# Send the initial field to the thermal solver.
#
# Each call to `.sub(j)` of a `Function` on a `MixedElement` seems to create a new copy.
# (Indeed, look at `dolfin.function.Function.sub` and `dolfin.function.Function.__init__`.)
# We need `FunctionAssigner` to set values on the original `Function`, so that the field
# does not vanish into a copy that is not used by the solver.
#
# https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
assigner = FunctionAssigner(thermal_solver.S, [V_rank0, V_rank0])  # FunctionAssigner(receiving_space, assigning_space)
assigner.assign(thermal_solver.s_n, [initial_T, initial_dTdt])  # old value: the actual initial condition
assigner.assign(thermal_solver.s_, [initial_T, initial_dTdt])  # latest Picard iterate: initial guess for new value
assigner.assign(thermal_solver.s_prev, [initial_T, initial_dTdt])  # previous Picard iterate, for convergence monitoring by user

# --------------------------------------------------------------------------------
# Thermal source term

# This represents cooling into the environment, in the direction perpendicular to the 2D sheet modeled.
# Heat flux through the 2D boundaries should be treated by a nonhomogeneous Neumann boundary condition instead.
#
# We could use a Stefan--Boltzmann radiative cooling law:
#
#    h = -r [T⁴ - T_ext⁴]
#
# But for simplicity, for now we use Newton's law of cooling:
#
#    h = -r [T - T_ext]
#
# In the internal energy balance equation, `h` is a specific heat source, [W/kg].
# The solver automatically multiplies it by `ρ` to obtain the volumetric heat source `ρ h`, [W/m³].
#
# Here `r` is a heat transfer coefficient. Dimension analysis yields [r] = W/(kg K).
# However, heat transfer coefficients are usually tabulated as [Γ] = W/(m² K).
# Thus we need a conversion factor with unit m²/kg, representing exposed area per unit mass.
#
# Consider a differential element of the sheet. We have:
#   dA = dx dy,                   exposed area when one side is exposed to air
#                                 (other side not included in dA, so it is perfectly insulated)
#   dm = ρ dx dy dz ≈ ρ dx dy H,  where H is the thickness of the sheet
# so
#   dA/dm = 1 / (ρ H)
# Here the  dx dy  cancels, so this ratio stays constant as dx → 0, dy → 0.
#
# You can make this model double-sided cooling simply by replacing `Γ → 2 * Γ` in config.py.
#
dAdm = 1 / (rho * H)   # [m²/kg]
r = dAdm * Γ  # [W/(kg K)]
def update_cooling():
    """Update the thermal source term according to Newton's law of cooling.

    The main loop calls this after each update of the temperature field.
    """
    thermal_solver.h_.assign(project(Constant(-r) * (fields["T"] - Constant(T_ext)), V_rank0))  # [W/m³]

    # # If using a data-based Neumann boundary condition, update also that with the latest data.
    # q_upper.assign(project(Constant(-Γ) * (fields["T"] - Constant(T_ext)), V_rank0))  # [W/m²]

# Set the value of the thermal source at the end of the first timestep.
update_cooling()
# Set it also at the beginning of the first timestep.
thermal_solver.h_n.assign(thermal_solver.h_)  # Not a mixed space, so we can copy like this (no temporaries).

# --------------------------------------------------------------------------------

# Enable stabilizers for the Galerkin formulation
linmom_solver.stabilizers.SUPG = enable_SUPG  # stabilizer for advection-dominant problems
thermal_solver.stabilizers.SUPG = enable_SUPG
SUPG_str = "[SUPG] " if enable_SUPG else ""  # for messages

mechanical_solver_str = "[mech] " if mechanical_solver_enabled else ""
thermal_solver_str = "[therm] " if thermal_solver_enabled else ""
solvers_str = f"{mechanical_solver_str}{thermal_solver_str}"

# https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
parameters['krylov_solver']['nonzero_initial_guess'] = True
# parameters['krylov_solver']['monitor_convergence'] = True  # DEBUG

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_v = XDMFFile(MPI.comm_world, vis_v_filename)
xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_dTdt = XDMFFile(MPI.comm_world, vis_dTdt_filename)
xdmffile_σ = XDMFFile(MPI.comm_world, vis_σ_filename)
xdmffile_εu = XDMFFile(MPI.comm_world, vis_εu_filename)
xdmffile_εv = XDMFFile(MPI.comm_world, vis_εv_filename)

# ParaView doesn't have a filter for von Mises stress, so we compute it ourselves.
# This is only for visualization.
xdmffile_vonMises = XDMFFile(MPI.comm_world, vis_vonMises_filename)
vonMises = Function(Q_rank0, name="vonMises")

for xdmffile in (xdmffile_u, xdmffile_v, xdmffile_T, xdmffile_dTdt,
                 xdmffile_σ, xdmffile_vonMises, xdmffile_εu, xdmffile_εv):
    xdmffile.parameters["flush_output"] = True
    xdmffile.parameters["rewrite_function_mesh"] = False
del xdmffile  # clean up loop counter from module-global scope

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_v = TimeSeries(sol_v_filename)
timeseries_T = TimeSeries(sol_T_filename)
timeseries_dTdt = TimeSeries(sol_dTdt_filename)
timeseries_σ = TimeSeries(sol_σ_filename)
timeseries_εu = TimeSeries(sol_εu_filename)
timeseries_εv = TimeSeries(sol_εv_filename)

# von Mises stress has no time series, because it's easily computed in FEniCS, given the stress field.

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# --------------------------------------------------------------------------------
# Prepare export

# HACK: Arrange things to allow exporting the velocity field at full nodal resolution.
all_V_rank1_dofs = np.array(range(V_rank1.dim()), "intc")
all_V_rank0_dofs = np.array(range(V_rank0.dim()), "intc")
all_Q_rank2_dofs = np.array(range(Q_rank2.dim()), "intc")
v_rank1_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V (tensor rank 1)
v_rank0_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V (tensor rank 0)
q_rank2_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on Q (tensor rank 2)

# TODO: We cannot export Q2 or Q3 quads at full nodal resolution in FEniCS 2019,
# TODO: because the mesh editor fails with "cell is not orderable".
#
# TODO: We could work around this on the unit square by just manually generating a suitable mesh.
# TODO: Right now we export only P2 or P3 triangles at full nodal resolution.
highres_export_V_rank1 = (V_rank1.ufl_element().degree() > 1 and V_rank1.ufl_element().family() == "Lagrange")
if highres_export_V_rank1:
    if my_rank == 0:
        print("Preparing export of higher-degree u/v data as refined P1...")
    with timer() as tim:
        v_rank1_P1, my_V_rank1_dofs = meshmagic.prepare_linear_export(V_rank1)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")
highres_export_V_rank0 = (V_rank0.ufl_element().degree() > 1 and V_rank0.ufl_element().family() == "Lagrange")
if highres_export_V_rank0:
    if my_rank == 0:
        print("Preparing export of higher-degree T/dTdt data as refined P1...")
    with timer() as tim:
        v_rank0_P1, my_V_rank0_dofs = meshmagic.prepare_linear_export(V_rank0)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")
highres_export_Q_rank2 = (Q_rank2.ufl_element().degree() > 1 and Q_rank2.ufl_element().family() == "Lagrange")
if highres_export_Q_rank2:
    if my_rank == 0:
        print("Preparing export of higher-degree σ data as refined P1...")
    with timer() as tim:
        q_rank2_P1, my_Q_rank2_dofs = meshmagic.prepare_linear_export(Q_rank2)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# --------------------------------------------------------------------------------
# Helper functions

def errnorm(u, u_prev, norm_type="h1"):
    """Error norm.

    Like `dolfin.errornorm`, but avoid using `dolfin.interpolate`, so that this
    can be called also when using quad elements.

    Note this implies we cannot use a higher-degree dG space to compute the norm,
    like `dolfin.errornorm` does, hence this won't be as accurate. But maybe this
    is enough for basic convergence monitoring of a system iteration.
    """
    V = u.function_space().collapse()
    e = Function(V)
    e.assign(project(u, V))  # TODO: can we use a `FunctionAssigner` to extract just one subfield?
    e.vector().axpy(-1.0, project(u_prev, V).vector())
    return norm(e, norm_type=norm_type, mesh=V.mesh())

def roundsig(x, significant_digits):
    """Round a float to a given number of significant digits."""
    # https://www.adamsmith.haus/python/answers/how-to-round-a-number-to-significant-digits-in-python
    import math
    digits_in_int_part = int(math.floor(math.log10(abs(x)))) + 1
    decimal_digits = significant_digits - digits_in_int_part
    return round(x, decimal_digits)

def elastic_strain_energy():
    """Form an UFL expression for elastic strain energy,  (1/2) σ : εel  [J].

    This automatically extracts the elastic strain εel from the total strain ε.
    """
    εth = linmom_solver.α(fields["T"]) * (fields["T"] - Constant(T0))
    εel = ε(fields["u"]) - εth
    return (1 / 2) * inner(fields["σ"], εel)
def kinetic_energy():
    """Form an UFL expression for kinetic energy,  (1/2) ρ v²  [J].

    Note the velocity is measured against the axially co-moving frame,
    so this is the kinetic energy seen by an observer in that frame.
    """
    # Note `linmom_solver._ρ`; we need the UFL `Constant` object here.
    return (1 / 2) * linmom_solver._ρ * dot(fields["du/dt"], fields["du/dt"])

def total_elastic_strain_energy():
    """Compute and return total elastic strain energy, ∫ (1/2) σ : εel dΩ  [J]."""
    return float(project(elastic_strain_energy(), W))  # project to ℝ (single global DOF)
def total_kinetic_energy():
    """Compute and return total kinetic energy, ∫ (1/2) ρ v² dΩ  [J]."""
    return float(project(kinetic_energy(), W))

# TODO: track and compute the total internal energy
#
# For constant specific heat capacity c, the total internal energy would be  ∫ ρ c T dΩ,
# but since actually  c = c(T),  we need to use a rate form, for which we don't currently
# track the history.
#
# We already have the data necessary to do this, though. Total internal energy at a
# material parcel is:
#   E = ρ e
# where, in this model, the density `ρ` is constant. The specific internal energy `e` is
#   e = ∫ de/dt t  (from t=0 to current time)
# Our constitutive model for internal energy is Joule's law:
#   e = c T
# Differentiating, the material rate of internal energy is
#   de/dt = d/dt (c T)
#         = dc/dt T + c dT/dt
#         = dc/dT dT/dt T + c dT/dt
#         = [dc/dT T + c] dT/dt
# All of these fields are readily available. This allows tracking the material rate
# (i.e. the actual physical rate) field of specific internal energy. Then we can just
# use the θ integrator, just like for all other fields here.
#
# Since the reference level of the internal energy is arbitrary, we can define the state
# at time t = 0 to be E = 0 in all of Ω.

# Preparation for plotting.
if my_rank == 0:
    print("Preparing plotter...")
with timer() as tim:
    # Analyze mesh and dofmap for plotting (slow; but static mesh, only need to do this once).
    #
    # The `Function` used for preparation MUST be defined on the SAME space as the `Function`
    # that will actually be plotted using that particular `prep`.
    #
    # For example, the space may be different for `u` and `v` even though both live on a
    # copy of `V`, because in a mixed space, these are different subspaces, so they have
    # different dofmaps.
    if my_rank == 0:
        print("    Computing visualization dofmaps...")

    prep_mixedV_rank1_subfield0_comp0 = plotmagic.mpiplot_prepare(fields["u"].sub(0))  # u_1
    prep_mixedV_rank1_subfield0_comp1 = plotmagic.mpiplot_prepare(fields["u"].sub(1))  # u_2
    prep_mixedV_rank1_subfield1_comp0 = plotmagic.mpiplot_prepare(fields["du/dt"].sub(0))  # v_1
    prep_mixedV_rank1_subfield1_comp1 = plotmagic.mpiplot_prepare(fields["du/dt"].sub(1))  # v_2

    prep_mixedV_rank0_subfield0 = plotmagic.mpiplot_prepare(fields["T"])
    prep_mixedV_rank0_subfield1 = plotmagic.mpiplot_prepare(fields["dT/dt"])

    # note also εu, εv have the same DOF structure (each of them also lives on Q_rank2)
    prep_Q_rank2_comp00 = plotmagic.mpiplot_prepare(fields["σ"].sub(0))
    prep_Q_rank2_comp01 = plotmagic.mpiplot_prepare(fields["σ"].sub(1))
    prep_Q_rank2_comp10 = plotmagic.mpiplot_prepare(fields["σ"].sub(2))
    prep_Q_rank2_comp11 = plotmagic.mpiplot_prepare(fields["σ"].sub(3))

    prep_Q_rank0 = plotmagic.mpiplot_prepare(Function(Q_rank0))
    prep_V_rank0 = plotmagic.mpiplot_prepare(Function(V_rank0))

    if my_rank == 0:
        print("    Creating figure window...")
        # NOTE: When using the OOP API of Matplotlib, it is important to **NOT** use
        # the `constrained_layout=True` option of `plt.subplots`; doing so will leak
        # plotting resources each time the figure is updated (it seems, especially
        # when colorbars are added?), making each plot drastically slower than the
        # previous one.
        #
        # Calling `plt.tight_layout()` manually (whenever the figure is updated)
        # avoids the resource leak.
        fig, axs = plt.subplots(3, 5, figsize=(12, 6))
        plt.tight_layout()
        plt.show()
        plt.draw()
        plotmagic.pause(0.001)
        colorbars = []
if my_rank == 0:
    print(f"Plotter preparation completed in {tim.dt:0.6g} seconds.")


def plotit():
    """Plot the current solution, updating the online visualization figure."""

    u_ = fields["u"]
    v_ = fields["du/dt"]
    T_ = fields["T"]
    dTdt_ = fields["dT/dt"]
    σ_ = fields["σ"]
    ε_ = fields["ε"]
    dεdt_ = fields["dε/dt"]

    def vrange(p):
        """Extract (min, max) from a scalar nodal FEM field."""
        minp, maxp = common.minmax(p, take_abs=False, mode="raw")
        return minp, maxp
    def symmetric_vrange(p):
        """Extract (-max, max) from the absolute value of a scalar nodal FEM field."""
        ignored_minp, maxp = common.minmax(p, take_abs=True, mode="raw")
        return -maxp, maxp

    def plot_one(field, prep, *, row, col, name, title, vrange_func, cmap="RdBu_r"):
        if my_rank == 0:
            print(f"DEBUG: plot {name}")
            ax = axs[row, col]
            ax.cla()
            plt.sca(ax)  # for `plotmagic.mpiplot`
        vmin, vmax = vrange_func(field)
        theplot = plotmagic.mpiplot(field, prep=prep, show_mesh=show_mesh,
                                    cmap=cmap, vmin=vmin, vmax=vmax)
        if my_rank == 0:
            print("DEBUG: colorbar")
            colorbars.append(fig.colorbar(theplot, ax=ax))
            ax.set_title(title)
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.set_aspect("equal")

    # remove old colorbars, since `ax.cla` doesn't
    if my_rank == 0:
        print("DEBUG: remove old colorbars")
        for cb in Popper(colorbars):
            cb.remove()

    plot_one(u_.sub(0), prep_mixedV_rank1_subfield0_comp0,
             row=0, col=0,
             name="u1", title=r"$u_{1}$ [m]", vrange_func=symmetric_vrange)
    plot_one(v_.sub(0), prep_mixedV_rank1_subfield1_comp0,
             row=0, col=1,
             name=f"v1 ≡ {dtext}u1/{dtext}t", title=f"$v_{{1}} \\equiv {dlatex} u_{{1}} / {dlatex} t$ [m/s]", vrange_func=symmetric_vrange)
    plot_one(u_.sub(1), prep_mixedV_rank1_subfield0_comp1,
             row=1, col=0,
             name="u2", title=r"$u_{2}$ [m]", vrange_func=symmetric_vrange)
    plot_one(v_.sub(1), prep_mixedV_rank1_subfield1_comp1,
             row=1, col=1,
             name=f"v2 ≡ {dtext}u2/{dtext}t", title=f"$v_{{2}} \\equiv {dlatex} u_{{2}} / {dlatex} t$ [m/s]", vrange_func=symmetric_vrange)

    if project_lower_degree_fields_to_V:
        # eliminate checkerboard pattern by postprocessing the dG0 function onto a C0 continuous space
        plot_one(project(ε_.sub(0), V_rank0), prep_V_rank0,
                 row=0, col=2,
                 name="ε11", title=r"$\varepsilon_{11}$", vrange_func=symmetric_vrange)
        plot_one(project(ε_.sub(1), V_rank0), prep_V_rank0,
                 row=1, col=2,
                 name="ε12", title=r"$\varepsilon_{12}$", vrange_func=symmetric_vrange)
        plot_one(project(ε_.sub(3), V_rank0), prep_V_rank0,
                 row=2, col=2,
                 name="ε22", title=r"$\varepsilon_{22}$", vrange_func=symmetric_vrange)
    else:
        plot_one(ε_.sub(0), prep_Q_rank2_comp00,
                 row=0, col=2,
                 name="ε11", title=r"$\varepsilon_{11}$", vrange_func=symmetric_vrange)
        plot_one(ε_.sub(1), prep_Q_rank2_comp01,
                 row=1, col=2,
                 name="ε12", title=r"$\varepsilon_{12}$", vrange_func=symmetric_vrange)
        # # ε21 = ε12, if the solver works correctly
        # plot_one(ε_.sub(2), prep_Q_rank2_comp10,
        #          row=XXX, col=XXX,
        #          name="ε21", title=r"$\varepsilon_{21}$", vrange_func=symmetric_vrange)
        plot_one(ε_.sub(3), prep_Q_rank2_comp11,
                 row=2, col=2,
                 name="ε22", title=r"$\varepsilon_{22}$", vrange_func=symmetric_vrange)

    if project_lower_degree_fields_to_V:
        plot_one(project(dεdt_.sub(0), V_rank0), prep_V_rank0,
                 row=0, col=3,
                 name=f"{dtext}ε11/{dtext}t", title=f"${dlatex} \\varepsilon_{{11}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
        plot_one(project(dεdt_.sub(1), V_rank0), prep_V_rank0,
                 row=1, col=3,
                 name=f"{dtext}ε12/{dtext}t", title=f"${dlatex} \\varepsilon_{{12}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
        plot_one(project(dεdt_.sub(3), V_rank0), prep_V_rank0,
                 row=2, col=3,
                 name=f"{dtext}ε22/{dtext}t", title=f"${dlatex} \\varepsilon_{{22}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
    else:
        plot_one(dεdt_.sub(0), prep_Q_rank2_comp00,
                 row=0, col=3,
                 name=f"{dtext}ε11/{dtext}t", title=f"${dlatex} \\varepsilon_{{11}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
        plot_one(dεdt_.sub(1), prep_Q_rank2_comp01,
                 row=1, col=3,
                 name=f"{dtext}ε12/{dtext}t", title=f"${dlatex} \\varepsilon_{{12}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
        # # dεdt21 = dεdt12, if the solver works correctly
        # plot_one(dεdt_.sub(2), prep_Q_rank2_comp10,
        #          row=XXX, col=XXX,
        #          name=f"{dtext}ε21/{dtext}t", title=f"${dlatex} \\varepsilon_{{21}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)
        plot_one(dεdt_.sub(3), prep_Q_rank2_comp11,
                 row=2, col=3,
                 name=f"{dtext}ε22/{dtext}t", title=f"${dlatex} \\varepsilon_{{22}} / {dlatex} t$ [1/s]", vrange_func=symmetric_vrange)

    if project_lower_degree_fields_to_V:
        plot_one(project(σ_.sub(0), V_rank0), prep_V_rank0,
                 row=0, col=4,
                 name="σ11", title=r"$\sigma_{11}$ [Pa]", vrange_func=symmetric_vrange)
        plot_one(project(σ_.sub(1), V_rank0), prep_V_rank0,
                 row=1, col=4,
                 name="σ12", title=r"$\sigma_{12}$ [Pa]", vrange_func=symmetric_vrange)
        plot_one(project(σ_.sub(3), V_rank0), prep_V_rank0,
                 row=2, col=4,
                 name="σ22", title=r"$\sigma_{22}$ [Pa]", vrange_func=symmetric_vrange)
    else:
        plot_one(σ_.sub(0), prep_Q_rank2_comp00,
                 row=0, col=4,
                 name="σ11", title=r"$\sigma_{11}$ [Pa]", vrange_func=symmetric_vrange)
        plot_one(σ_.sub(1), prep_Q_rank2_comp01,
                 row=1, col=4,
                 name="σ12", title=r"$\sigma_{12}$ [Pa]", vrange_func=symmetric_vrange)
        # # σ21 = σ12, if the solver works correctly
        # plot_one(σ_.sub(2), prep_Q_rank2_comp10,
        #          row=XXX, col=XXX,
        #          name="σ21", title=r"$\sigma_{21}$ [Pa]", vrange_func=symmetric_vrange)
        plot_one(σ_.sub(3), prep_Q_rank2_comp11,
                 row=2, col=4,
                 name="σ22", title=r"$\sigma_{22}$ [Pa]", vrange_func=symmetric_vrange)

    if thermal_solver_enabled:
        # We actually plot the difference to the reference temperature, to be able to judge heating/cooling easily.
        T_minus_T0 = project(T_ - Constant(T0), V_rank0)
        plot_one(T_minus_T0, prep_V_rank0,
                 row=2, col=0,
                 name="T - T0", title=r"$T - T_0$ [K]", vrange_func=symmetric_vrange)
        plot_one(dTdt_, prep_mixedV_rank0_subfield1,
                 row=2, col=1,
                 name=f"{dtext}T/{dtext}t", title=f"${dlatex} T / {dlatex} t$ [K/s]", vrange_func=symmetric_vrange)
    else:
        if project_lower_degree_fields_to_V:
            E = project(elastic_strain_energy(), V_rank0)
            plot_one(E, prep_V_rank0,
                     row=2, col=0,
                     name="elastic strain energy", title=r"$(1/2) \sigma : \varepsilon_{\mathrm{el}}$ [J/m³]", vrange_func=vrange, cmap="viridis")
        else:
            E = project(elastic_strain_energy(), Q_rank0)
            plot_one(E, prep_Q_rank0,
                     row=2, col=0,
                     name="elastic strain energy", title=r"$(1/2) \sigma : \varepsilon_{\mathrm{el}}$ [J/m³]", vrange_func=vrange, cmap="viridis")
        K = project(kinetic_energy(), V_rank0)  # kinetic energy (as seen by observer in axially co-moving frame)
        plot_one(K, prep_V_rank0,
                 row=2, col=1,
                 name="kinetic energy", title=r"$(1/2) \rho v^2$ [J/m³]", vrange_func=vrange, cmap="viridis")

    # figure title (progress message)
    if my_rank == 0:
        print("DEBUG: update figure title")
        fig.suptitle(msg)

    if my_rank == 0:
        print("DEBUG: render plot")
        plt.tight_layout()
        # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
        plotmagic.pause(0.001)
        print("DEBUG: plotting done")


def export_fields(u_, v_, T_, dTdt_, σ_, εu_, εv_, *, t):
    """Export solution fields to `.xdmf`."""

    if highres_export_V_rank1:
        # Save the displacement visualization at full nodal resolution.
        #
        # But because `u` and `du/dt` live on the same mixed space, we must extract the subfields first
        # into separate fields on two non-mixed spaces.
        #
        # FunctionAssigner(receiving_space, assigning_space)
        u_tmp = Function(V_rank1)
        v_tmp = Function(V_rank1)
        assigner = FunctionAssigner([u_tmp.function_space(),
                                     v_tmp.function_space()],
                                    linmom_solver.S)  # mixed function space of mechanical solver, containing both u and du/dt
        assigner.assign([u_tmp, v_tmp], linmom_solver.s_)

        u_tmp.vector().gather(v_rank1_vec_copy, all_V_rank1_dofs)  # allgather `u_tmp` to `v_rank1_vec_copy`
        v_rank1_P1.vector()[:] = v_rank1_vec_copy[my_V_rank1_dofs]  # LHS MPI-local; RHS global
        v_rank1_P1.rename(fields["u"].name(), "a Function")
        xdmffile_u.write(v_rank1_P1, t)

        # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
        v_tmp.vector().gather(v_rank1_vec_copy, all_V_rank1_dofs)  # allgather `v_tmp` to `v_rank1_vec_copy`
        v_rank1_P1.vector()[:] = v_rank1_vec_copy[my_V_rank1_dofs]  # LHS MPI-local; RHS global
        v_rank1_P1.rename(fields["du/dt"].name(), "a Function")
        xdmffile_v.write(v_rank1_P1, t)
    else:  # save at P1 resolution
        xdmffile_u.write(u_, t)
        xdmffile_v.write(v_, t)

    if highres_export_V_rank0:
        # Same here - extract `T` and `dT/dt` from the mixed space.
        T_tmp = Function(V_rank0)
        dTdt_tmp = Function(V_rank0)
        assigner = FunctionAssigner([T_tmp.function_space(),
                                     dTdt_tmp.function_space()],
                                    thermal_solver.S)  # mixed function space of thermal solver, containing both T and dT/dt
        assigner.assign([T_tmp, dTdt_tmp], thermal_solver.s_)

        # Save the displacement visualization at full nodal resolution.
        T_tmp.vector().gather(v_rank0_vec_copy, all_V_rank0_dofs)  # allgather `T_tmp` to `v_rank0_vec_copy`
        v_rank0_P1.vector()[:] = v_rank0_vec_copy[my_V_rank0_dofs]  # LHS MPI-local; RHS global
        v_rank0_P1.rename(fields["T"].name(), "a Function")
        xdmffile_T.write(v_rank0_P1, t)

        # `dT/dt` lives on a copy of the same function space as `T`; recycle the temporary vector
        dTdt_tmp.vector().gather(v_rank0_vec_copy, all_V_rank0_dofs)  # allgather `dTdt_` to `v_rank0_vec_copy`
        v_rank0_P1.vector()[:] = v_rank0_vec_copy[my_V_rank0_dofs]  # LHS MPI-local; RHS global
        v_rank0_P1.rename(fields["dT/dt"].name(), "a Function")
        xdmffile_dTdt.write(v_rank0_P1, t)
    else:  # save at P1 resolution
        xdmffile_T.write(T_, t)
        xdmffile_dTdt.write(dTdt_, t)

    # The fields that live on Q live on a non-mixed space, so here we don't need any special processing.
    if highres_export_Q_rank2:
        σ_.vector().gather(q_rank2_vec_copy, all_Q_rank2_dofs)
        q_rank2_P1.vector()[:] = q_rank2_vec_copy[my_Q_rank2_dofs]
        q_rank2_P1.rename(fields["σ"].name(), "a Function")
        xdmffile_σ.write(q_rank2_P1, t)
    else:  # save at P1 resolution
        xdmffile_σ.write(σ_, t)

    if highres_export_Q_rank2:
        εu_.vector().gather(q_rank2_vec_copy, all_Q_rank2_dofs)
        q_rank2_P1.vector()[:] = q_rank2_vec_copy[my_Q_rank2_dofs]
        q_rank2_P1.rename(fields["ε"].name(), "a Function")
        xdmffile_εu.write(q_rank2_P1, t)
    else:  # save at P1 resolution
        xdmffile_εu.write(εu_, t)

    if highres_export_Q_rank2:
        εv_.vector().gather(q_rank2_vec_copy, all_Q_rank2_dofs)
        q_rank2_P1.vector()[:] = q_rank2_vec_copy[my_Q_rank2_dofs]
        q_rank2_P1.rename(fields["dε/dt"].name(), "a Function")
        xdmffile_εv.write(q_rank2_P1, t)
    else:  # save at P1 resolution
        xdmffile_εv.write(εv_, t)

    # compute von Mises stress for visualization in ParaView
    # TODO: export von Mises stress at full nodal resolution, too
    #
    # TODO: von Mises stress in 2D - does this argument make sense?
    #
    # The deviatoric part is *defined* as the traceless part, so for a 2D tensor the
    # factor appearing in `dev` is (1/2), not (1/3). The motivation of the definition
    # of the von Mises stress is to scale the representative stress √(s:s) by a factor
    # that makes it match the stress when in uniaxial tension. See e.g.:
    #     https://www.continuummechanics.org/vonmisesstress.html
    #
    # In 2D, we have
    #     σ = [[σ11 0] [0 0]]  (uniaxial tension; pure 2D case, not embedded in 3D)
    #     d = 2   (dimension)
    #     tr(σ) ≡ ∑ σkk = σ11
    #     s := dev(σ) ≡ σ - (1/d) I tr(σ)
    #                 = σ - (1/2) I tr(σ)
    #                 = [[(1/2)*σ11 0] [0 -(1/2)*σ11]]
    #     s:s ≡ ∑ ski ski = (1/4) σ11² + (1/4) σ11² = (1/2) σ11²
    #     σ_rep = √(s:s) = √(1/2) σ11
    # To match the uniaxial stress, we define
    #     σ_VM2D := √(2) σ_rep = σ11
    # so the scaling factor appearing in the definition of a pure-2D von Mises stress
    # is found to be √(2).
    #
    # Note this is for pure 2D (where both stress and strain are 2D; no third dimension
    # exists), not 3D under plane stress. For the latter, we would use the standard 3D
    # formulas as-is.
    #
    def dev(tensor):
        """Deviatoric (traceless) part of a rank-2 tensor.

        This is the true traceless part, using a scaling factor of `1 / d`,
        where `d` is the geometric dimensionality, not a hard-coded `1 / 3`.
        """
        d = tensor.geometric_dimension()
        return tensor - (1 / d) * tr(tensor) * Identity(d)
    s_ = dev(σ_)
    d = s_.geometric_dimension()
    dim_to_scale_factor = {3: sqrt(3 / 2), 2: sqrt(2)}
    scale = dim_to_scale_factor[d]
    vonMises_expr = scale * sqrt(inner(s_, s_))
    vonMises.assign(project(vonMises_expr, Q_rank0))
    xdmffile_vonMises.write(vonMises, t)

# --------------------------------------------------------------------------------
# Compute dynamic solution

# FunctionAssigner(receiving_space, assigning_space)
assigner = FunctionAssigner([linmom_solver.T_.function_space(),
                             linmom_solver.dTdt_.function_space()],
                            thermal_solver.S)  # mixed function space of thermal solver, containing both T and dT/dt

# Send in initial fields to the external field inputs.
# We use here the fact that the latest field values are initialized to the initial values.
assigner.assign([linmom_solver.T_n, linmom_solver.dTdt_n], thermal_solver.s_)
assigner.assign([linmom_solver.T_, linmom_solver.dTdt_], thermal_solver.s_)

linmom_solver.export_stress()
thermal_solver.σ_n.assign(linmom_solver.σ_)  # old value
thermal_solver.σ_.assign(thermal_solver.σ_n)  # initial guess for new value
thermal_solver.a_n.assign(project(linmom_solver.a + linmom_solver.v_,
                                  thermal_solver.a_n.function_space()))
thermal_solver.a_.assign(thermal_solver.a_n)

t = 0
vis_count = 0
msg = "Starting. Progress information will be available shortly..."
vis_step_walltime_local = 0
nsavemod = max(1, int(nt / nsave_total))  # every how manyth timestep to save
nvismod = max(1, int(vis_every * nt))  # every how manyth timestep to visualize
est = ETAEstimator(nt, keep_last=nvismod)
if my_rank == 0:
    print(f"Saving max. {nsave_total} timesteps in total -> save every {nsavemod} timestep{'s' if nsavemod > 1 else ''}.")
    nvisualizations = round(1 / vis_every)
    print(f"Visualizing at every {100.0 * vis_every:0.3g}% of simulation ({nvisualizations} visualization{'s' if nvisualizations > 1 else ''} total) -> vis every {nvismod} timestep{'s' if nvismod > 1 else ''}.")
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Update value in time-dependent boundary conditions, if any
    for expr in (u0_left, u0_right):
        if expr:
            expr.u0 = u0_func(t)
    for expr in (σ0_left, σ0_right):
        if expr:
            expr.σ0 = σ0_func(t)

    # Solve one timestep.
    # Multiphysics problem with weak coupling between subproblems.
    n_system_iterations = 0
    converged = False
    while not converged:
        n_system_iterations += 1

        if mechanical_solver_enabled:
            # Mechanical substep
            linmom_solver.step()

        if thermal_solver_enabled:
            # Send updated external fields to thermal solver.
            # Cauchy stress.
            #
            # NOTE: If only the thermal solver is enabled, the material will seem to experience a nonzero isotropic stress state,
            # although its total strain is zero.
            #
            # This is caused by the thermal strain. Because the total displacement is zero (mechanical solver disabled), the Cauchy stress
            # calculation yields, correctly, that there must be an elastic strain that exactly cancels the thermal expansion/contraction,
            # thereby making the total strain zero. It is this elastic strain that produces the Cauchy stress.
            #
            # In other words, when only the thermal solver is enabled, the Cauchy stress field shows how much stress would have to be applied
            # to keep the total strain at zero (i.e. to cancel the thermal expansion/contraction).
            #
            linmom_solver.export_stress()
            thermal_solver.σ_.assign(linmom_solver.σ_)
            # Advection velocity.
            # NOTE: The thermal solver needs material parcel velocity with respect to the *laboratory* frame.
            # NOTE: This is the axial velocity, plus the material parcel velocity with respect to the *co-moving* frame.
            thermal_solver.a_.assign(project(linmom_solver.a + linmom_solver.v_,
                                             thermal_solver.a_.function_space()))

            # Thermal substep
            thermal_solver.step()

            # Update cooling term for next iteration
            update_cooling()

        if mechanical_solver_enabled:
            # Send updated external fields to mechanical solver
            # Could do this:
            #     linmom_solver.T_.assign(project(thermal_solver.s_.sub(0),
            #                                     linmom_solver.T_.function_space()))
            #     linmom_solver.dTdt_.assign(project(thermal_solver.s_.sub(1),
            #                                        linmom_solver.dTdt_.function_space()))
            # But there's a more civilized way - use a FunctionAssigner:
            assigner.assign([linmom_solver.T_, linmom_solver.dTdt_], thermal_solver.s_)

        # Monitor the convergence of the system iteration.
        H1_diffs = {"u": errnorm(linmom_solver.s_.sub(0), linmom_solver.s_prev.sub(0), "h1"),
                    "du/dt": errnorm(linmom_solver.s_.sub(1), linmom_solver.s_prev.sub(1), "h1"),
                    "T": errnorm(thermal_solver.s_.sub(0), thermal_solver.s_prev.sub(0), "h1"),
                    "dT/dt": errnorm(thermal_solver.s_.sub(1), thermal_solver.s_prev.sub(1), "h1")}
        if my_rank == 0:
            print(f"    timestep {n + 1}, system iteration {n_system_iterations}, ‖u - u_prev‖_H1 = {H1_diffs['u']:0.6g}, ‖du/dt - du/dt_prev‖_H1 = {H1_diffs['du/dt']:0.6g}, ‖T - T_prev‖_H1 = {H1_diffs['T']:0.6g}, ‖dT/dt - dT/dt_prev‖_H1 = {H1_diffs['dT/dt']:0.6g}")
        if all(H1_diff < H1_tol for H1_diff in H1_diffs.values()):
            if my_rank == 0:
                print(f"    timestep {n + 1}, system converged after iteration {n_system_iterations}")
            converged = True
        if n_system_iterations > maxit:
            raise RuntimeError(f"    timestep {n + 1}, system did not converge after {maxit} system iterations. Simulation terminated.")

    # Converged. Accept the timestep.
    # This updates the "old" solution and the "old" external fields, and initializes the Picard iterate for the next timestep
    # to the accepted solution of this timestep.
    linmom_solver.commit()
    thermal_solver.commit()

    # Export and visualize
    u_ = fields["u"]
    v_ = fields["du/dt"]
    T_ = fields["T"]
    dTdt_ = fields["dT/dt"]
    σ_ = fields["σ"]
    εu_ = fields["ε"]
    εv_ = fields["dε/dt"]

    if n % nsavemod == 0 or n == nt - 1:
        begin("Saving")
        export_fields(u_, v_, T_, dTdt_, σ_, εu_, εv_, t=t)
        timeseries_u.store(u_.vector(), t)  # the timeseries saves the original data
        timeseries_v.store(v_.vector(), t)
        timeseries_T.store(T_.vector(), t)
        timeseries_dTdt.store(dTdt_.vector(), t)
        timeseries_σ.store(σ_.vector(), t)
        timeseries_εu.store(εu_.vector(), t)
        timeseries_εv.store(εv_.vector(), t)
        end()

    end()

    # Plot the components of u
    visualize = n % nvismod == 0 or n == nt - 1
    if visualize:
        begin("Plotting")
        with timer() as tim:
            plotit()
            # # info for msg (expensive; only update these once per vis step)
            # # No space for these in the suptitle; leaving them out.
            # minu, maxu = common.minmax(u_, mode="l2")
            # minT, maxT = common.minmax(T_, mode="l2")

            # magnitude of advection velocity, for Courant and Péclet numbers
            maga = project(mag(thermal_solver.a_), V_rank0)

            # maximum advection velocity, for Péclet number
            maxa_local = np.array(maga.vector()).max()
            maxa_global = MPI.comm_world.allgather(maxa_local)
            maxa = max(maxa_global)

            # Courant number (advection of `T` in thermal solver)
            Co_adv = project(maga * Constant(dt) / thermal_solver.he, V_rank0)
            maxCo_local = np.array(Co_adv.vector()).max()
            maxCo_global = MPI.comm_world.allgather(maxCo_local)
            maxCo_thermal = max(maxCo_global)

            # Péclet number (ratio of advective vs. diffusive effects) of thermal solver, rough approximation.
            if maxa != 0.0:
                d = thermal_solver.s_.geometric_dimension()
                ν = project(((1 / d) * tr(thermal_solver.k(T_))) / (thermal_solver.ρ * thermal_solver.c(T_)), V_rank0)  # diffusivity
                minν_local = np.array(ν.vector()).min()
                minν_global = MPI.comm_world.allgather(minν_local)
                minν = min(minν_global)
                domain_length = xmax - xmin  # characteristic length (TODO: parameterize this)
                # If minν = 0, the division will emit a "RuntimeWarning: divide by zero encountered in double_scalars".
                # Under these circumstances the correct answer is indeed Pe = ∞, so ignore the warning.
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore",
                                            message="^divide by zero .*$",
                                            category=RuntimeWarning,
                                            module="__main__")
                    maxPe_thermal = maxa * domain_length / minν
            else:
                maxPe_thermal = 0.0

            # Courant number (advection of `v` in linear momentum solver)
            # The velocity field in the advection operator is the axial drive field (V0, 0).
            # v_el = project(sqrt(E_func(T_) / Constant(rho)), V_rank0)  # elastic wave propagation speed, irrelevant here
            Co_mech = project(Constant(V0) * Constant(dt) / linmom_solver.he, V_rank0)
            maxCo_local = np.array(Co_mech.vector()).max()
            maxCo_global = MPI.comm_world.allgather(maxCo_local)
            maxCo_mech = max(maxCo_global)

            # Péclet number, advection vs. diffusion of `v`, rough approximation.
            # Here the velocity field is the axial drive velocity.
            if V0 != 0.0:
                maxa = V0
                λ = linmom_solver.λ(T_)
                μ = linmom_solver.μ(T_)
                τ = linmom_solver._τ  # the UFL Constant object
                ν = project(τ * Minn(λ, 2 * μ) / Constant(rho), V_rank0)  # representative diffusivity of velocity, τ ‖E‖ / ρ
                minν_local = np.array(ν.vector()).min()
                minν_global = MPI.comm_world.allgather(minν_local)
                minν = min(minν_global)
                domain_length = xmax - xmin  # characteristic length; TODO: parameterize this
                # If τ = 0, then minν = 0, and the division will emit a
                # "RuntimeWarning: divide by zero encountered in double_scalars".
                # Under these circumstances the correct answer is indeed Pe = ∞, so ignore the warning.
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore",
                                            message="^divide by zero .*$",
                                            category=RuntimeWarning,
                                            module="__main__")
                    maxPe_mech = maxa * domain_length / minν
            else:
                maxPe_mech = 0.0

            # maximum in-domain cooling rate [W/m²]
            maxh_local = -1.0 * thermal_solver.h_.vector().min() * rho * H
            maxh_global = MPI.comm_world.allgather(maxh_local)
            maxh = max(maxh_global)

        last_plot_walltime_local = tim.dt
        last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
        last_plot_walltime = max(last_plot_walltime_global)
        end()

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()
    # TODO: make dt, dt_avg part of the public interface in `unpythonic`
    dt_avg = sum(est.que) / len(est.que)
    vis_step_walltime_local = nvismod * dt_avg

    E = total_elastic_strain_energy()
    K = total_kinetic_energy()
    if my_rank == 0:  # DEBUG
        print(f"Timestep {n + 1}/{nt} ({100 * (n + 2) / nt:0.1f}%); t = {t + dt:0.6g}; Δt = {dt:0.6g}; Pe_th = {maxPe_thermal:0.2g}; Co_th = {maxCo_thermal:0.2g}; Pe_mech = {maxPe_mech:0.2g}; Co_mech = {maxCo_mech:0.2g}; max cooling rate = {maxh:0.2g} W/m²; E = ∫ (1/2) σ:εel dΩ = {E:0.3g}; K = ∫ (1/2) ρ v² dΩ = {K:0.3g}; wall time per timestep {dt_avg:0.3g}s; avg {1/dt_avg:0.3g} timesteps/sec (running avg, n = {len(est.que)})")

    # In MPI mode, one of the worker processes may have a larger slice of the domain
    # (or require more Krylov iterations to converge) than the root process.
    # So to get a reliable ETA, we must take the maximum across all processes.
    # But MPI communication is expensive, so only update this at vis steps.
    if visualize:
        times_global = MPI.comm_world.allgather((vis_step_walltime_local, est.estimate, est.formatted_eta))
        item_with_max_estimate = max(times_global, key=lambda item: item[1])
        max_eta = item_with_max_estimate[2]
        item_with_max_vis_step_walltime = max(times_global, key=lambda item: item[0])
        max_vis_step_walltime = item_with_max_vis_step_walltime[0]

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{solvers_str}{SUPG_str}{n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n_system_iterations} iterations; $\\mathrm{{Pe}}_\\mathrm{{th}}$ = {maxPe_thermal:0.2g}; $\\mathrm{{Co}}_\\mathrm{{th}}$ = {maxCo_thermal:0.2g}; $\\mathrm{{Pe}}_\\mathrm{{mech}}$ = {maxPe_mech:0.2g}; $\\mathrm{{Co}}_\\mathrm{{mech}}$ = {maxCo_mech:0.2g}; V₀ = {V0} m/s; τ = {tau:0.3g} s; vis every {roundsig(max_vis_step_walltime, 2):g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

    # Loop-and-a-half situation, so draw one more time to update title.
    if visualize and my_rank == 0:
        # figure title (progress message)
        fig.suptitle(msg)
        plt.tight_layout()
        # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
        plotmagic.pause(0.001)
        plt.savefig(f"{fig_output_dir}{fig_basename}{vis_count:06d}.{fig_format}")
        vis_count += 1

# Hold plot
if my_rank == 0:
    print("Simulation complete.")
    plt.ioff()
    plt.show()
