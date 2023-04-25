# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of axial motion).

Thermomechanical variant; models also thermal expansion.
"""

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
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import (LinearMomentumBalance,
                                InternalEnergyBalance)
from extrafeathers.pdes.eulerian_solid_advanced import ε
from extrafeathers.pdes.numutil import mag
from .config import (rho, tau, V0, T0, Γ, T_ext, H, dt, nt, H1_tol, maxit,
                     lamda_func, mu_func, α_func, dαdT_func, c_func, dcdT_func, k_func,
                     nsave_total, vis_every, enable_SUPG, show_mesh,
                     Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_v_filename, sol_v_filename,
                     vis_T_filename, sol_T_filename,
                     vis_dTdt_filename, sol_dTdt_filename,
                     vis_σ_filename, sol_σ_filename,
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
#
# Also note that this choice would not satisfy the relevant inf-sup condition
# were we to use a mixed method to solve the linear momentum balance.
# But we actually solve using a primal formulation, and Q is used only
# for visualizing results, so it's ok.
#
# The solvers use a mixed function space, built from copies of these spaces:
#   - `u` is allocated one `V_rank1` space, and `du/dt` is allocated another.
#   - `T` is allocated one `V_rank0` space, and `dT/dt` is allocated another.
#
# The rank-0 spaces are also used in visualization, when there is a need to L2-project
# a scalar expression into a FEM space suitable for plotting.
#
V_rank1 = VectorFunctionSpace(mesh, 'P', 1)
V_rank0 = V_rank1.sub(0).collapse()
Q_rank2 = TensorFunctionSpace(mesh, 'DP', 0)
Q_rank0 = Q_rank2.sub(0).collapse()

# Start by detecting the bounding box - this can be used e.g. for fixing the
# displacement on a line inside the domain.
with timer() as tim:
    ignored_cells, nodes_dict = meshmagic.all_cells(V_rank0)
    ignored_dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])
if my_rank == 0:
    print(f"Geometry detection completed in {tim.dt:0.6g} seconds.")
    print(f"x ∈ [{xmin:0.6g}, {xmax:0.6g}], y ∈ [{ymin:0.6g}, {ymax:0.6g}].")

# --------------------------------------------------------------------------------
# Set up the solvers

# The stress uses a Neumann BC, with the boundary stress field set here.
# The stress field given here is evaluated (and projected into the outer normal direction)
# on the boundaries that have no Dirichlet boundary condition on `u`.
bcσ = [(Constant(((1e8, 0), (0, 0))), None)]  # [Pa]

# The heat flux uses uses a Neumann BC, with the boundary scalar flux
# (in the direction of the outer normal) set here.
bcq = [(Constant(0), None)]  # [W/m²]

# Dirichlet boundary condition lists. We create the lists now, because the solver constructor needs to store a reference to the list instance.
# But to actually set up Dirichlet BCs, we need a reference to the subspaces created by solver initialization. So we will fill these lists later.
bcu = []
bcT = []

# Instantiate the solvers for our multiphysics problem.
# We use backward Euler time integration (θ = 1) to help stabilize the numerics.
linmom_solver = LinearMomentumBalance(V_rank1, Q_rank2, Q_rank2,
                                      rho, lamda_func, mu_func, tau,
                                      α_func, dαdT_func, T0,
                                      V0,
                                      bcu, bcσ, dt, θ=1.0)
thermal_solver = InternalEnergyBalance(V_rank0,
                                       rho, c_func, dcdT_func, k_func, T0,
                                       bcT, bcq, dt, θ=1.0,
                                       advection="general", use_stress=True)

# Plotting labels for the rate operator. This model uses the advective rate "d/dt".
#   - The displacement solver is mixed Lagrangean-Eulerian (MLE); the advection velocity is the axial drive velocity.
#     The displacement `u` and the material parcel velocity `du/dt` are measured against the axially co-moving frame.
#   - The thermal solver is pure Eulerian; the advection velocity is the full velocity of the material parcels in the laboratory frame.
#     The temperature rate `dT/dt` is the *material* derivative of the temperature `T`.
dlatex = r"\mathrm{d}"
dtext = "d"

if my_rank == 0:
    print(f"Number of DOFs: u {V_rank1.dim()}, {dtext}u/{dtext}t {V_rank1.dim()}, T {V_rank0.dim()}, {dtext}T/{dtext}t {V_rank0.dim()}")

# NOTE: Accessing the `.sub(j)` of a mixed field (e.g. `s_` here) seems to create a new copy of the subfield every time.
# So e.g. `fields["T"]` gives read access; but to write to a mixed field, we must use a `FunctionAssigner`. Examples further below.
fields = {"u": linmom_solver.s_.sub(0),
          "du/dt": linmom_solver.s_.sub(1),
          "T": thermal_solver.s_.sub(0),
          "dT/dt": thermal_solver.s_.sub(1),
          "σ": linmom_solver.σ_,
          "ε": linmom_solver.εu_,
          "dε/dt": linmom_solver.εv_}
subspaces = {k: v.function_space() for k, v in fields.items()}  # for setting boundary conditions

# --------------------------------------------------------------------------------
# Dirichlet boundary conditions, mechanical subproblem

# In our examples the initial field for `u` is zero, which is also the default.

# These are used only by cases with a time-dependent boundary condition on `u`,
# but the main loop expects the variables to exist (to detect whether to use them or not).
u0_left = None
u0_right = None
u0_func = lambda t: 0.0

# Here are some examples of how to make a time-dependent boundary condition for `u`:
#
# # Left and right edges: fixed left end, displacement-controlled pull at right end
# bcu_left = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
# from fenics import Expression
# u0_func = lambda t: 1e-2 * t
# u0_right = Expression(("u0", "0"), degree=1, u0=u0_func(0.0))
# bcu_right = DirichletBC(subspaces["u"], u0_right, boundary_parts, Boundaries.RIGHT.value)
# bcu.append(bcu_left)
# bcu.append(bcu_right)

# # Left and right edges: fixed left end, displacement-controlled *u1 only* at right end
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

# Left and right edges: fixed left end, stress-controlled pull at right end (Kurki et al. 2016).
bcu_left = DirichletBC(subspaces["u"], Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
bcu.append(bcu_left)

# --------------------------------------------------------------------------------
# Dirichlet boundary conditions, thermal subproblem

T_left = T0
# T_right = T0

# Axially moving continuum: specify the temperature of the material parcels that enter the domain at the left.
# Don't set anything at the right - the default zero Neumann (no change in temperature in axial direction i.e. steady outflow) is appropriate.
bcT_left = DirichletBC(subspaces["T"], Constant(T_left), boundary_parts, Boundaries.LEFT.value)
bcT.append(bcT_left)

# --------------------------------------------------------------------------------
# Initial conditions, thermal subproblem

# Our domain is Ω = (-0.5, 0.5)².

# A linear function of x is at least a trivial steady-state solution of the standard heat equation, so we can use something like that.
# from fenics import Expression
# initial_T = project(Expression("T_left + (T_right - T_left) * (x[0] + 0.5)", degree=1, T_left=T_left, T_right=T_right), V_rank0)
initial_T = project(Constant(T_left), V_rank0)
initial_dTdt = Function(V_rank0)  # zeroes

# Send the initial field to the thermal solver.
#
# Each call to `.sub(j)` of a `Function` on a `MixedElement` seems to create a new copy.
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
# For simplicity, we use Newton's law of cooling:
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
dAdm = 1 / (rho * H)   # [m²/kg]
r = dAdm * Γ  # [W/(kg K)]
def update_cooling():
    """Update the thermal source term according to Newton's law of cooling.

    The main loop calls this after each update of the temperature field.
    """
    thermal_solver.h_.assign(project(Constant(-r) * (fields["T"] - Constant(T_ext)), V_rank0))  # [W/m³]

# Set the value of the thermal source at the end of the first timestep.
update_cooling()
# Set it also at the beginning of the first timestep.
thermal_solver.h_n.assign(thermal_solver.h_)  # Not a mixed space, so we can copy like this (no temporaries).

# --------------------------------------------------------------------------------

# Enable stabilizers for the Galerkin formulation
linmom_solver.stabilizers.SUPG = enable_SUPG  # stabilizer for advection-dominant problems
thermal_solver.stabilizers.SUPG = enable_SUPG
SUPG_str = "[SUPG] " if enable_SUPG else ""  # for messages

# https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
parameters['krylov_solver']['nonzero_initial_guess'] = True
# parameters['krylov_solver']['monitor_convergence'] = True  # DEBUG

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_v = XDMFFile(MPI.comm_world, vis_v_filename)
xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_dTdt = XDMFFile(MPI.comm_world, vis_dTdt_filename)
xdmffile_σ = XDMFFile(MPI.comm_world, vis_σ_filename)

# ParaView doesn't have a filter for von Mises stress, so we compute it ourselves.
# This is only for visualization.
xdmffile_vonMises = XDMFFile(MPI.comm_world, vis_vonMises_filename)
vonMises = Function(Q_rank0)

for xdmffile in (xdmffile_u, xdmffile_v, xdmffile_T, xdmffile_dTdt, xdmffile_σ, xdmffile_vonMises):
    xdmffile.parameters["flush_output"] = True
    xdmffile.parameters["rewrite_function_mesh"] = False
del xdmffile  # clean up loop counter from module-global scope

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_v = TimeSeries(sol_v_filename)
timeseries_T = TimeSeries(sol_T_filename)
timeseries_dTdt = TimeSeries(sol_dTdt_filename)
timeseries_σ = TimeSeries(sol_σ_filename)

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

W = FunctionSpace(mesh, "R", 0)  # Function space of ℝ (single global DOF)
def elastic_strain_energy():
    """Compute and return total elastic strain energy, ∫ (1/2) σ : εel dΩ  [J].

    This automatically extracts the elastic strain εel from the total strain ε.
    """
    εth = linmom_solver.α(fields["T"]) * (fields["T"] - Constant(T0))
    εel = ε(fields["u"]) - εth
    return float(project((1 / 2) * inner(fields["σ"], εel), W))
def kinetic_energy():
    """Compute and return total kinetic energy, ∫ (1/2) ρ v² dΩ  [J].

    Note the velocity is measured against the axially co-moving frame,
    so this is the kinetic energy seen by an observer in that frame.
    """
    # Note `linmom_solver._ρ`; we need the UFL `Constant` object here.
    return float(project((1 / 2) * linmom_solver._ρ * dot(fields["du/dt"], fields["du/dt"]), W))

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

    # We actually plot the difference to the reference temperature, to be able to judge heating/cooling easily.
    T_minus_T0 = project(T_ - Constant(T0), V_rank0)
    plot_one(T_minus_T0, prep_V_rank0,
             row=2, col=0,
             name="T - T0", title=r"$T - T_0$ [K]", vrange_func=symmetric_vrange)
    plot_one(dTdt_, prep_mixedV_rank0_subfield1,
             row=2, col=1,
             name=f"{dtext}T/{dtext}t", title=f"${dlatex} T / {dlatex} t$ [K/s]", vrange_func=symmetric_vrange)

    # TODO: Do we want an energy visualization? If so, rethink plot layout.
    # # In the original pure mechanical variant, we used to have 13 plots, but 15 subplot slots,
    # # so we used the last two to plot the energy. But the thermomechanical variant of the model has 15 plots.
    # # Could also be useful to see the thermal and mechanical strains separately.
    # εth = linmom_solver.α(T_) * (T_ - Constant(T0))
    # εel = ε(u_) - εth
    # E = project((1 / 2) * inner(σ_, εel), Q_rank0)  # elastic strain energy
    # plot_one(E, prep_Q_rank0,
    #          row=2, col=0,
    #          name="elastic strain energy", title=r"$(1/2) \sigma : \varepsilon_{\mathrm{el}}$ [J/m³]", vrange_func=vrange, cmap="viridis")
    # K = project((1 / 2) * linmom_solver._ρ * dot(v_, v_), V_rank0)  # kinetic energy (as seen by observer in axially co-moving frame)
    # plot_one(K, prep_V_rank0,
    #          row=2, col=1,
    #          name"kinetic energy", title=r"$(1/2) \rho v^2$ [J/m³]", vrange_func=vrange, cmap="viridis")

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


def export_fields(u_, v_, T_, dTdt_, σ_, *, t):
    """Export solution fields to `.xdmf`."""

    if highres_export_V_rank1:
        # Save the displacement visualization at full nodal resolution.
        u_.vector().gather(v_rank1_vec_copy, all_V_rank1_dofs)  # allgather `u_` to `v_rank1_vec_copy`
        v_rank1_P1.vector()[:] = v_rank1_vec_copy[my_V_rank1_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(v_rank1_P1, t)

        # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
        v_.vector().gather(v_rank1_vec_copy, all_V_rank1_dofs)  # allgather `v_` to `v_rank1_vec_copy`
        v_rank1_P1.vector()[:] = v_rank1_vec_copy[my_V_rank1_dofs]  # LHS MPI-local; RHS global
        xdmffile_v.write(v_rank1_P1, t)
    else:  # save at P1 resolution
        xdmffile_u.write(u_, t)
        xdmffile_v.write(v_, t)

    if highres_export_V_rank0:
        # Save the displacement visualization at full nodal resolution.
        T_.vector().gather(v_rank0_vec_copy, all_V_rank0_dofs)  # allgather `T_` to `v_rank0_vec_copy`
        v_rank0_P1.vector()[:] = v_rank0_vec_copy[my_V_rank0_dofs]  # LHS MPI-local; RHS global
        xdmffile_T.write(v_rank0_P1, t)

        # `dT/dt` lives on a copy of the same function space as `T`; recycle the temporary vector
        dTdt_.vector().gather(v_rank0_vec_copy, all_V_rank0_dofs)  # allgather `dTdt_` to `v_rank0_vec_copy`
        v_rank0_P1.vector()[:] = v_rank0_vec_copy[my_V_rank0_dofs]  # LHS MPI-local; RHS global
        xdmffile_dTdt.write(v_rank0_P1, t)
    else:  # save at P1 resolution
        xdmffile_T.write(T_, t)
        xdmffile_dTdt.write(dTdt_, t)

    if highres_export_Q_rank2:
        σ_.vector().gather(q_rank2_vec_copy, all_Q_rank2_dofs)
        q_rank2_P1.vector()[:] = q_rank2_vec_copy[my_Q_rank2_dofs]
        xdmffile_σ.write(q_rank2_P1, t)
    else:  # save at P1 resolution
        xdmffile_σ.write(σ_, t)

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
    def dev(T):
        """Deviatoric (traceless) part of rank-2 tensor `T`.

        This is the true traceless part, taking the dimensionality of `T` as-is.
        """
        d = T.geometric_dimension()
        return T - (1 / d) * tr(T) * Identity(d)
    s = dev(σ_)
    d = s.geometric_dimension()
    dim_to_scale_factor = {3: sqrt(3 / 2), 2: sqrt(2)}
    scale = dim_to_scale_factor[d]
    vonMises_expr = scale * sqrt(inner(s, s))
    vonMises.assign(project(vonMises_expr, Q_rank0))
    xdmffile_vonMises.write(vonMises, t)

# --------------------------------------------------------------------------------
# Compute dynamic solution

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

    # Solve one timestep.
    # Multiphysics problem with weak coupling between subproblems.
    n_system_iterations = 0
    converged = False
    while not converged:
        n_system_iterations += 1

        # Mechanical substep
        linmom_solver.step()

        # Send updated external fields to thermal solver.
        # Cauchy stress.
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

    if n % nsavemod == 0 or n == nt - 1:
        begin("Saving")
        export_fields(u_, v_, T_, dTdt_, σ_, t=t)
        timeseries_u.store(u_.vector(), t)  # the timeseries saves the original data
        timeseries_v.store(v_.vector(), t)
        timeseries_T.store(T_.vector(), t)
        timeseries_dTdt.store(dTdt_.vector(), t)
        timeseries_σ.store(σ_.vector(), t)
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

            # Courant number
            Co_adv = project(maga * Constant(dt) / thermal_solver.he, V_rank0)
            maxCo_local = np.array(Co_adv.vector()).max()
            maxCo_global = MPI.comm_world.allgather(maxCo_local)
            maxCo = max(maxCo_global)

            # Péclet number (ratio of advective vs. diffusive effects), rough approximation.
            d = thermal_solver.s_.geometric_dimension()
            ν = project(((1 / d) * tr(thermal_solver.k(T_))) / (thermal_solver.ρ * thermal_solver.c(T_)), V_rank0)  # diffusivity
            minν_local = np.array(ν.vector()).min()
            minν_global = MPI.comm_world.allgather(minν_local)
            minν = min(minν_global)
            L = xmax - xmin  # characteristic length; here we use the domain length (TODO: parameterize this)
            maxPe = maxa * L / minν

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

    E = elastic_strain_energy()
    K = kinetic_energy()
    if my_rank == 0:  # DEBUG
        print(f"Timestep {n + 1}/{nt} ({100 * (n + 2) / nt:0.1f}%); t = {t + dt:0.6g}; Δt = {dt:0.6g}; Pe = {maxPe:0.2g}; Co = {maxCo:0.2g}; max cooling rate = {maxh:0.2g} W/m²; E = ∫ (1/2) σ:εel dΩ = {E:0.3g}; K = ∫ (1/2) ρ v² dΩ = {K:0.3g}; K + E = {K + E:0.3g}; wall time per timestep {dt_avg:0.3g}s; avg {1/dt_avg:0.3g} timesteps/sec (running avg, n = {len(est.que)})")

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
    msg = f"{SUPG_str}{n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n_system_iterations} iterations; Pe = {maxPe:0.2g}; Co = {maxCo:0.2g}; V₀ = {V0} m/s; τ = {tau:0.3g} s; vis every {roundsig(max_vis_step_walltime, 2):g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

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
