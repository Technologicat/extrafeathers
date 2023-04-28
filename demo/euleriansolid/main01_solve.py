# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of axial motion)."""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer, Popper

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
                    DirichletBC,
                    Constant, Function,
                    project, Vector,
                    tr, Identity, sqrt, inner, dot,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end,
                    parameters)

# custom utilities for FEniCS
from extrafeathers import common
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import (EulerianSolid,  # noqa: F401
                                EulerianSolidAlternative,
                                EulerianSolidPrimal,
                                step_adaptive,
                                SteadyStateEulerianSolid,
                                SteadyStateEulerianSolidAlternative,
                                SteadyStateEulerianSolidPrimal)
from extrafeathers.pdes.eulerian_solid import ε
from .config import (rho, lamda, mu, tau, V0, dt, nt,
                     dynamic, nsave_total, vis_every, enable_SUPG, show_mesh,
                     Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_v_filename, sol_v_filename,
                     vis_σ_filename, sol_σ_filename,
                     vis_vonMises_filename,
                     fig_output_dir, fig_basename, fig_format)


my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 1)  # displacement
Q = TensorFunctionSpace(mesh, 'P', 2)  # stress

# Scalar function spaces with the same element family and degree as `V` and `Q`, for visualization purposes.
Vscalar = V.sub(0).collapse()
Qscalar = Q.sub(0).collapse()

# Start by detecting the bounding box - we need this in some examples for fixing the
# displacement on a line inside the domain.
with timer() as tim:
    ignored_cells, nodes_dict = meshmagic.all_cells(Vscalar)
    ignored_dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])
if my_rank == 0:
    print(f"Geometry detection completed in {tim.dt:0.6g} seconds.")
    print(f"x ∈ [{xmin:0.6g}, {xmax:0.6g}], y ∈ [{ymin:0.6g}, {ymax:0.6g}].")

# --------------------------------------------------------------------------------
# Choose the solver

# TODO: Fix multi-headed hydra (dynamic and steady-state cases currently interleaved in one script).
# TODO: The mess is even worse now that we have several alternative algorithms available for each case.

bcu = []  # for steady-state solver
bcv = []  # for dynamic solver
bcσ = []  # for both solvers (except primal solvers, which use a Neumann BC for the stress)
if dynamic:
    # Straightforward Eulerian formulation, with `v := ∂u/∂t`.
    #
    # `P`: function space for strain projection, before inserting the strain into the constitutive law.
    #
    # - Discontinuous strain spaces seem to give rise to numerical oscillations in the stress.
    #   Thus, to stabilize, it is important to choose an appropriate continuous space here.
    #   Which spaces are "appropriate" is left as an exercise to the reader.
    # - It seems a degree-1 space is too small to give correct results.
    #   This is likely related to the size requirements for the stress space.
    #
    # # P = TensorFunctionSpace(self.mesh, "DG", 1)  # oscillations along `a` (like old algo without strain projection)
    # # P = TensorFunctionSpace(self.mesh, "DG", 2)  # oscillations along `a` (like old algo without strain projection)
    # # P = TensorFunctionSpace(self.mesh, Q.ufl_element().family(), 1)  # results completely wrong
    # P = Q  # Q2; just small oscillations near high gradients of `u` and `v`
    # solver = EulerianSolid(V, Q, P, rho, lamda, mu, tau, V0, bcv, bcσ, dt)  # Crank-Nicolson (default)
    # # solver = EulerianSolid(V, Q, P, rho, lamda, mu, tau, V0, bcv, bcσ, dt, θ=1.0)  # backward Euler
    # # Set plotting labels; this formulation uses v := ∂u/∂t
    # dlatex = r"\partial"
    # dtext = "∂"

    # # Alternative formulation, with `v := du/dt`.
    # # Only uses the space `P` for visualizing the strains.
    # P = TensorFunctionSpace(mesh, 'DP', 0)
    # solver = EulerianSolidAlternative(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcv, bcσ, dt)  # Crank-Nicolson (default)
    # # solver = EulerianSolidAlternative(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcv, bcσ, dt, θ=1.0)  # backward Euler
    # # Set plotting labels; this formulation uses v := du/dt
    # dlatex = r"\mathrm{d}"
    # dtext = "d"

    # Primal formulation (`u` and `v` only), with `v := du/dt`.
    # Only uses the space `P` for visualizing the strains.
    # The stress uses a Neumann BC, with the boundary stress field set here.
    # The stress field given here is evaluated on the boundaries that have
    # no Dirichlet boundary condition on `u`.
    P = TensorFunctionSpace(mesh, 'DP', 0)
    boundary_stress = Constant(((1e6, 0), (0, 0)))
    solver = EulerianSolidPrimal(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcv, boundary_stress, dt)
    # Set plotting labels; this formulation uses v := du/dt
    dlatex = r"\mathrm{d}"
    dtext = "d"

else:  # steady state
    # The steady-state solvers only use the space `P` for visualizing the strains.
    P = TensorFunctionSpace(mesh, 'DP', 0)

    # # Straightforward Eulerian formulation.
    # # NOTE: This algorithm does not work yet.
    # solver = SteadyStateEulerianSolid(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcσ)
    # # Set plotting labels; this formulation uses v := ∂u/∂t
    # dlatex = r"\partial"
    # dtext = "∂"

    # # Alternative formulation, with `v := du/dt = (a·∇)u` (last equality holds because steady state).
    # # NOTE: This algorithm does not work yet.
    # solver = SteadyStateEulerianSolidAlternative(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcv, bcσ)
    # # Set plotting labels; this formulation uses v := du/dt
    # dlatex = r"\mathrm{d}"
    # dtext = "d"

    # Primal formulation (`u` and `v` only), with `v := du/dt = (a·∇)u` (last equality because steady state).
    # The stress uses a Neumann BC, with the boundary stress field set here.
    # The stress field given here is evaluated on the boundaries that have
    # no Dirichlet boundary condition on `u`.
    boundary_stress = Constant(((1e6, 0), (0, 0)))
    solver = SteadyStateEulerianSolidPrimal(V, Q, P, rho, lamda, mu, tau, V0, bcu, bcv, boundary_stress)
    # Set plotting labels; this formulation uses v := du/dt
    dlatex = r"\mathrm{d}"
    dtext = "d"

if my_rank == 0:
    print(f"Number of DOFs: velocity {V.dim()}, strain {P.dim()}, stress {Q.dim()}")

# Adapter: where each solver stores its solution fields
fields = {EulerianSolid: {"u": lambda solver: solver.u_,
                          "v": lambda solver: solver.v_,
                          "σ": lambda solver: solver.σ_},
          SteadyStateEulerianSolid: {"u": lambda solver: solver.s_.sub(0),
                                     "v": lambda solver: solver.v_,  # unused, all zeros
                                     "σ": lambda solver: solver.s_.sub(1)},
          EulerianSolidAlternative: {"u": lambda solver: solver.u_,
                                     "v": lambda solver: solver.v_,
                                     "σ": lambda solver: solver.σ_},
          SteadyStateEulerianSolidAlternative: {"u": lambda solver: solver.s_.sub(0),
                                                "v": lambda solver: solver.s_.sub(1),
                                                "σ": lambda solver: solver.s_.sub(2)},
          EulerianSolidPrimal: {"u": lambda solver: solver.s_.sub(0),
                                "v": lambda solver: solver.s_.sub(1),
                                "σ": lambda solver: solver.σ_},
          SteadyStateEulerianSolidPrimal: {"u": lambda solver: solver.s_.sub(0),
                                           "v": lambda solver: solver.s_.sub(1),
                                           "σ": lambda solver: solver.σ_}}
Usubspace = fields[type(solver)]["u"](solver).function_space()
Vsubspace = fields[type(solver)]["v"](solver).function_space()
Qsubspace = fields[type(solver)]["σ"](solver).function_space()

# For setting initial fields in dynamic solvers only
if dynamic:
    oldfields = {EulerianSolid: {"u": lambda solver: solver.u_n,
                                 "v": lambda solver: solver.v_n,
                                 "σ": lambda solver: solver.σ_n},
                 EulerianSolidAlternative: {"u": lambda solver: solver.u_n,
                                            "v": lambda solver: solver.v_n,
                                            "σ": lambda solver: solver.σ_n},
                 EulerianSolidPrimal: {"u": lambda solver: solver.s_n.sub(0),
                                       "v": lambda solver: solver.s_n.sub(1),
                                       "σ": lambda solver: NotImplemented}}
else:
    oldfields = None

# --------------------------------------------------------------------------------
# Define boundary conditions

# NOTE: The primal solvers, which use the mixed space, do not use Dirichlet BCs for `σ`,
# but instead use a Neumann BC (which we already set up as `boundary_stress` when
# instantiating the solver).

# --------------------------------------------------------------------------------
# Boundary conditions for dynamic solver

# These are needed only by cases with a time-dependent boundary condition on `u`.
u0_left = None
u0_right = None
u0_func = lambda t: 0.0

if dynamic:
    # In all of the examples, we set the top and bottom edges the same way:

    # Top and bottom edges: zero normal stress
    #
    # Need `method="geometric"` to detect boundary DOFs on discontinuous spaces.
    # Important for the mixed methods, since in theory they can use a discontinuous
    # space for the stress variable.
    #
    # This is missing from the latest docs; see old docs.
    #   https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/fem/bcs/DirichletBC.html
    #   https://fenicsproject.org/olddocs/dolfin/latest/python/_autogenerated/dolfin.cpp.fem.html#dolfin.cpp.fem.DirichletBC
    #
    # From the 1.3.0 docs linked above; quoted here for preservation:
    #     The ‘method’ variable may be used to specify the type of method used to identify degrees of freedom
    #     on the boundary. Available methods are: topological approach (default), geometric approach, and
    #     pointwise approach. The topological approach is faster, but will only identify degrees of freedom
    #     that are located on a facet that is entirely on the boundary. In particular, the topological
    #     approach will not identify degrees of freedom for discontinuous elements (which are all internal to
    #     the cell). A remedy for this is to use the geometric approach. In the geometric approach, each dof
    #     on each facet that matches the boundary condition will be checked. To apply pointwise boundary
    #     conditions e.g. pointloads, one will have to use the pointwise approach which in turn is the
    #     slowest of the three possible methods. The three possibilties are “topological”, “geometric” and
    #     “pointwise”.
    bcσ_top1 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ12 (symm.)
    bcσ_top2 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ21
    bcσ_top3 = DirichletBC(Qsubspace.sub(3), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ22
    bcσ_bottom1 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ12
    bcσ_bottom2 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ21
    bcσ_bottom3 = DirichletBC(Qsubspace.sub(3), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ22
    bcσ.append(bcσ_top1)
    bcσ.append(bcσ_top2)
    bcσ.append(bcσ_top3)
    bcσ.append(bcσ_bottom1)
    bcσ.append(bcσ_bottom2)
    bcσ.append(bcσ_bottom3)

    # # Left and right edges: fixed displacement
    # #
    # # `EulerianSolid` takes no BCs for `u` (which is simply the time integral of `v`);
    # # instead, set an initial condition on `u`, and set `v = 0` at the fixed boundaries.
    # #
    # # The other algorithms expect BCs on `u` instead of `v`.
    # #
    # # Note that a sudden nonzero fixed displacement at the edges comes as quite a shock.
    # # The solver might not converge, if the initial condition for `u` is far from a physically
    # # valid state. Furthermore, for some initial states, Kelvin-Voigt might converge, but
    # # linear elastic might not.
    # #
    # from fenics import Expression
    # # # u0 = project(Expression(("1e-3 * 2.0 * (x[0] - 0.5)", "0"), degree=1), V)  # [0, 1]²
    # u0 = project(Expression(("1e-3 * 2.0 * x[0]", "0"), degree=1), V)  # [-0.5, 0.5]²
    # # # u0 = project(Expression(("1e-3 * 2.0 * x[0]",
    # # #                          f"-{ν} * 1e-3 * 2.0 * x[1] * 2.0 * pow((0.5 - abs(x[0])), 0.5)"),
    # # #                         degree=1),
    # # #              V)  # [-0.5, 0.5]²
    # # TODO: monolithic variants need a `FunctionAssigner` to avoid assigning to an unused copy.
    # oldfields[type(solver)]["u"](solver).assign(u0)
    # fields[type(solver)]["u"](solver).assign(u0)
    # # NOTE: `v` is defined as either `∂u/∂t` or `du/dt`, depending on the solver used.
    # bcv_left = DirichletBC(Vsubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # bcv_right = DirichletBC(Vsubspace, Constant((0, 0)), boundary_parts, Boundaries.RIGHT.value)
    # bcv.append(bcv_left)
    # bcv.append(bcv_right)

    # The following examples are designed for `EulerianSolidAlternative` and `EulerianSolidPrimal`,
    # as these algorithms in general perform better. Especially recommended is `EulerianSolidPrimal`,
    # which is the fastest of the provided dynamic solvers, and also A-stable.
    #
    # In these examples the initial field for `u` is zero, so it does not need to be specified.

    # # Left and right edges: fixed left end, displacement-controlled pull at right end
    # from fenics import Expression
    # u0_func = lambda t: 1e-2 * t
    # bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # u0_right = Expression(("u0", "0"), degree=1, u0=u0_func(0.0))
    # bcu_right = DirichletBC(Usubspace, u0_right, boundary_parts, Boundaries.RIGHT.value)
    # bcu.append(bcu_left)
    # bcu.append(bcu_right)

    # # Left and right edges: fixed left end, displacement-controlled *u1 only* at right end
    # # TODO: For now, this example needs `EulerianSolidPrimal`. Figure out which components of `σ`
    # # TODO: should be set at the right. Maybe just `σ12` and `σ21`?
    # from fenics import Expression
    # u0_func = lambda t: 1e-2 * t
    # bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # u0_right = Expression("u0", degree=1, u0=u0_func(0.0))
    # bcu_right = DirichletBC(Usubspace.sub(0), u0_right, boundary_parts, Boundaries.RIGHT.value)  # u1
    # bcu.append(bcu_left)
    # bcu.append(bcu_right)

    # # Left and right edges: displacement-controlled pull
    # from fenics import Expression
    # u0_func = lambda t: 1e-2 * t
    # # `dolfin.Expression` compiles to C++, so we must define these separately. Trying to flip the sign
    # # of an `Expression` and setting that to a `DirichletBC` causes a one-time `project` to take place.
    # # That won't even work here, but even if it did, it wouldn't give us an updatable.
    # u0_left = Expression(("-u0", "0"), degree=1, u0=u0_func(0.0))
    # u0_right = Expression(("+u0", "0"), degree=1, u0=u0_func(0.0))
    # bcu_left = DirichletBC(Usubspace, u0_left, boundary_parts, Boundaries.LEFT.value)
    # bcu_right = DirichletBC(Usubspace, u0_right, boundary_parts, Boundaries.RIGHT.value)
    # bcu.append(bcu_left)
    # bcu.append(bcu_right)

    # Left and right edges: fixed left end, stress-controlled pull at right end (Kurki et al. 2016).
    bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    bcu.append(bcu_left)
    bcσ_right1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ11
    bcσ_right2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ12
    bcσ_right3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ21 (symm.)
    bcσ.append(bcσ_right1)
    bcσ.append(bcσ_right2)
    bcσ.append(bcσ_right3)

    # # Left and right edges: stress-controlled pull at both ends
    # # TODO: does not work yet as-is, rigid-body mode remover needs work
    # bcσ_left1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ11
    # bcσ_left2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ12
    # bcσ_left3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ21 (symm.)
    # bcσ_right1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ11
    # bcσ_right2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ12
    # bcσ_right3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ21 (symm.)
    # bcσ.append(bcσ_left1)
    # bcσ.append(bcσ_left2)
    # bcσ.append(bcσ_left3)
    # bcσ.append(bcσ_right1)
    # bcσ.append(bcσ_right2)
    # bcσ.append(bcσ_right3)
    # # TODO: Obviously, this works if we fix the displacement somewhere...
    # # bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # # bcu.append(bcu_left)
    # # TODO: ...so let's fix the displacement at the center. Note this needs the regular grid
    # # TODO: to work properly, because we need to have DOFs on the center line.
    # # https://fenicsproject.org/qa/10273/pointwise-bc/
    # from fenics import CompiledSubDomain
    # xmid = (xmin + xmax) / 2
    # ymid = (ymin + ymax) / 2
    # center_vline = CompiledSubDomain(f"near(x[0], {xmid})")
    # # center_hline = CompiledSubDomain(f"near(x[1], {ymid})")
    # center_point = CompiledSubDomain(f"near(x[0], {xmid}) && near(x[1], {ymid})")
    # bcu_center1 = DirichletBC(Usubspace.sub(0), Constant(0), center_vline, method="pointwise")  # u1(0, y) = 0
    # bcu_center2 = DirichletBC(Usubspace.sub(1), Constant(0), center_point, method="pointwise")  # u2(0, 0) = 0
    # bcu.append(bcu_center1)
    # bcu.append(bcu_center2)

# --------------------------------------------------------------------------------
# Boundary conditions for steady-state solver

if not dynamic:
    # In all of the examples, we set the top and bottom edges the same way:

    # Top and bottom edges: zero normal stress
    bcσ_top1 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ12 (symm.)
    bcσ_top2 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ21
    bcσ_top3 = DirichletBC(Qsubspace.sub(3), Constant(0), boundary_parts, Boundaries.TOP.value, "geometric")  # σ22
    bcσ_bottom1 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ12
    bcσ_bottom2 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ21
    bcσ_bottom3 = DirichletBC(Qsubspace.sub(3), Constant(0), boundary_parts, Boundaries.BOTTOM.value, "geometric")  # σ22
    bcσ.append(bcσ_top1)
    bcσ.append(bcσ_top2)
    bcσ.append(bcσ_top3)
    bcσ.append(bcσ_bottom1)
    bcσ.append(bcσ_bottom2)
    bcσ.append(bcσ_bottom3)

    # # Left and right edges: fixed displacement
    # #
    # # NOTE: In `SteadyStateEulerianSolidPrimal`, `v = du/dt`.
    # #
    # # At first glance, this looks like the linear first-order transport PDE... but actually,
    # # since the equation defines `v` (with `u` determined from the linear momentum balance),
    # # the equation is just an L2 projection of `du/dt`. Therefore, `v` takes no BCs.
    #
    # # If `v = du/dt` was instead taken to define `u` (with `v` determined from other considerations),
    # # *then* the equation would need BCs for `u` on the inflow boundary.
    # bcu_left = DirichletBC(Usubspace, Constant((-1e-3, 0)), boundary_parts, Boundaries.LEFT.value)
    # bcu_right = DirichletBC(Usubspace, Constant((+1e-3, 0)), boundary_parts, Boundaries.RIGHT.value)
    # # bcv_left = DirichletBC(Vsubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # # bcv_right = DirichletBC(Vsubspace, Constant((0, 0)), boundary_parts, Boundaries.RIGHT.value)
    # bcu.append(bcu_left)
    # bcu.append(bcu_right)
    # # bcv.append(bcv_left)
    # # bcv.append(bcv_right)

    # Left and right edges: fixed left end, stress-controlled pull at right end (Kurki et al. 2016).
    bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    bcu.append(bcu_left)
    # bcv_left = DirichletBC(Vsubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # bcv.append(bcv_left)
    bcσ_right1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ11
    bcσ_right2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ12
    bcσ_right3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ21 (symm.)
    bcσ.append(bcσ_right1)
    bcσ.append(bcσ_right2)
    bcσ.append(bcσ_right3)

    # # Left and right edges: stress-controlled pull at both ends
    # # TODO: Does not work yet as-is, rigid-body mode remover needs work.
    # bcσ_left1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ11
    # bcσ_left2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ12
    # bcσ_left3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.LEFT.value, "geometric")  # σ21 (symm.)
    # bcσ_right1 = DirichletBC(Qsubspace.sub(0), Constant(1e6), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ11
    # bcσ_right2 = DirichletBC(Qsubspace.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ12
    # bcσ_right3 = DirichletBC(Qsubspace.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value, "geometric")  # σ21 (symm.)
    # bcσ.append(bcσ_left1)
    # bcσ.append(bcσ_left2)
    # bcσ.append(bcσ_left3)
    # bcσ.append(bcσ_right1)
    # bcσ.append(bcσ_right2)
    # bcσ.append(bcσ_right3)
    # # TODO: Obviously, this works if we fix the displacement somewhere...
    # # bcu_left = DirichletBC(Usubspace, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
    # # bcu.append(bcu_left)
    # # TODO: ...so let's fix the displacement at the center. Note this needs the regular grid
    # # TODO: to work properly, because we need to have DOFs on the center line.
    # # https://fenicsproject.org/qa/10273/pointwise-bc/
    # from fenics import CompiledSubDomain
    # xmid = (xmin + xmax) / 2
    # ymid = (ymin + ymax) / 2
    # center_vline = CompiledSubDomain(f"near(x[0], {xmid})")
    # # center_hline = CompiledSubDomain(f"near(x[1], {ymid})")
    # center_point = CompiledSubDomain(f"near(x[0], {xmid}) && near(x[1], {ymid})")
    # bcu_center1 = DirichletBC(Usubspace.sub(0), Constant(0), center_vline, method="pointwise")  # u1(0, y) = 0
    # bcu_center2 = DirichletBC(Usubspace.sub(1), Constant(0), center_point, method="pointwise")  # u2(0, 0) = 0
    # bcu.append(bcu_center1)
    # bcu.append(bcu_center2)

    # TODO: update this
    # # Set nonzero initial guess for `u`
    # from fenics import Expression
    # from .config import ν
    # u0 = project(Expression(("1e-3 * 2.0 * x[0]",
    #                          f"-{ν} * 1e-3 * 2.0 * x[1] * 2.0 * pow((0.5 - abs(x[0])), 0.5)"),
    #                         degree=1),
    #              Usubspace)  # [-0.5, 0.5]²
    # # fields[type(solver)]["u"](solver).assign(u0)

    # TODO: update this
    # # Set nonzero initial guess for `σ`
    # σ = fields[type(solver)]["σ"](solver)  # each call seems to create a new copy?
    # # (Indeed, look at `dolfin.function.Function.sub` and `dolfin.function.Function.__init__`.)
    # σ11 = σ.sub(0)
    # # from fenics import Expression
    # # σ.assign(project(Expression((("1e6 * cos(2 * pi * x[0])", 0), (0, 0)), degree=2), Qsubspace.collapse()))  # DEBUG testing
    # # σ.assign(project(Constant(((1e6, 0), (0, 0))), Qsubspace.collapse()))
    # # σ.assign(project(Constant(((1.0, 2.0), (3.0, 4.0))), Qsubspace.collapse()))
    # σ11.assign(project(Constant(1e6), Qsubspace.sub(0).collapse()))
    # # theplot = plotmagic.mpiplot(σ11)
    # # plt.colorbar(theplot)
    # # plt.show()
    # # crash

    # TODO: update this
    # # To set the IG reliably in the monolithic case:
    # # Each call to `.sub(j)` of a `Function` on a `MixedElement` seems to create a new copy.
    # # (Indeed, look at `dolfin.function.Function.sub` and `dolfin.function.Function.__init__`.)
    # # We need `FunctionAssigner` to set values on the original `Function`, so that the field
    # # does not vanish into a copy that is not used by the solver.
    # from fenics import Expression, FunctionAssigner
    # from .config import ν
    # # FunctionAssigner(receiving_space, assigning_space)
    # assigner = FunctionAssigner(solver.S, [V, V, Q])  # for `SteadyStateEulerianSolidAlternative`
    # # assigner = FunctionAssigner(solver.S, [V, Q])  # for `SteadyStateEulerianSolid`
    # zeroV = Function(V)
    # # u0 = project(Expression(("1e-3 * (2.0 * x[0])", "0"), degree=1), V)  # [-0.5, 0.5]²
    # # u0 = project(Expression(("1e-3 * 2.0 * x[0]",
    # #                          f"-{ν} * 1e-3 * (2.0 * x[1]) * pow((1.0 - abs(2.0 * x[0])), 0.5)"),
    # #                         degree=1),
    # #              V)  # [-0.5, 0.5]²
    # u0 = project(Expression(("1e-6 * (0.5 + x[0])", f"-{ν} * 1e-6 * (2.0 * x[1])"), degree=1), V)
    # σ0 = project(Constant(((1e6, 0), (0, 0))), Q)
    # assigner.assign(solver.s_, [u0, zeroV, σ0])  # for `SteadyStateEulerianSolidAlternative`
    # # assigner.assign(solver.s_, [u0, σ0])  # for `SteadyStateEulerianSolid`
    # # σ = fields[type(solver)]["σ"](solver)
    # # σ11 = σ.sub(0)
    # # theplot = plotmagic.mpiplot(σ11)
    # # plt.colorbar(theplot)
    # # plt.show()
    # # crash

# --------------------------------------------------------------------------------

# Enable stabilizers for the Galerkin formulation
#
# NOTE: Some of the solvers provide SUPG, while others (mainly those where it did not work or help) do not.
# NOTE: The figure window will nevertheless always have the "[SUPG]" indicator when the SUPG setting is on.
solver.stabilizers.SUPG = enable_SUPG  # stabilizer for advection-dominant problems
SUPG_str = "[SUPG] " if enable_SUPG else ""  # for messages

# https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
parameters['krylov_solver']['nonzero_initial_guess'] = True
# parameters['krylov_solver']['monitor_convergence'] = True  # DEBUG

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_v = XDMFFile(MPI.comm_world, vis_v_filename)
xdmffile_v.parameters["flush_output"] = True
xdmffile_v.parameters["rewrite_function_mesh"] = False

xdmffile_σ = XDMFFile(MPI.comm_world, vis_σ_filename)
xdmffile_σ.parameters["flush_output"] = True
xdmffile_σ.parameters["rewrite_function_mesh"] = False

# ParaView doesn't have a filter for von Mises stress, so we compute it ourselves.
xdmffile_vonMises = XDMFFile(MPI.comm_world, vis_vonMises_filename)
xdmffile_vonMises.parameters["flush_output"] = True
xdmffile_vonMises.parameters["rewrite_function_mesh"] = False
vonMises = Function(Qscalar)

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_v = TimeSeries(sol_v_filename)
timeseries_σ = TimeSeries(sol_σ_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# --------------------------------------------------------------------------------
# Prepare export

# HACK: Arrange things to allow exporting the velocity field at full nodal resolution.
all_V_dofs = np.array(range(V.dim()), "intc")
all_Q_dofs = np.array(range(Q.dim()), "intc")
v_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
q_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on Q

# TODO: We cannot export Q2 or Q3 quads at full nodal resolution in FEniCS 2019,
# TODO: because the mesh editor fails with "cell is not orderable".
#
# TODO: We could work around this on the unit square by just manually generating a suitable mesh.
# TODO: Right now we export only P2 or P3 triangles at full nodal resolution.
highres_export_V = (V.ufl_element().degree() > 1 and V.ufl_element().family() == "Lagrange")
if highres_export_V:
    if my_rank == 0:
        print("Preparing export of higher-degree u/v data as refined P1...")
    with timer() as tim:
        v_P1, my_V_dofs = meshmagic.prepare_linear_export(V)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")
highres_export_Q = (Q.ufl_element().degree() > 1 and Q.ufl_element().family() == "Lagrange")
if highres_export_Q:
    if my_rank == 0:
        print("Preparing export of higher-degree σ data as refined P1...")
    with timer() as tim:
        q_P1, my_Q_dofs = meshmagic.prepare_linear_export(Q)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# --------------------------------------------------------------------------------
# Helper functions

def roundsig(x, significant_digits):
    # https://www.adamsmith.haus/python/answers/how-to-round-a-number-to-significant-digits-in-python
    import math
    digits_in_int_part = int(math.floor(math.log10(abs(x)))) + 1
    decimal_digits = significant_digits - digits_in_int_part
    return round(x, decimal_digits)

# W = FunctionSpace(mesh, "DP", 0)  # dG0 space
# def elastic_strain_energy():
#     E = project((1 / 2) * inner(solver.σ_, ε(solver.u_)),
#                 W,
#                 form_compiler_parameters={"quadrature_degree": 2})
#     return np.sum(E.vector()[:])
W = FunctionSpace(mesh, "R", 0)  # Function space of ℝ (single global DOF)
def elastic_strain_energy():
    "Compute and return total elastic strain energy, ∫ (1/2) σ : ε dΩ"
    return float(project((1 / 2) * inner(solver.σ_, ε(solver.u_)), W))
def kinetic_energy():
    "Compute and return total kinetic energy, ∫ (1/2) ρ v² dΩ"
    # Note `solver._ρ`; we need the UFL `Constant` object here.
    return float(project((1 / 2) * solver._ρ * dot(solver.v_, solver.v_), W))

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

    tmp = fields[type(solver)]["u"](solver)
    prep_U0 = plotmagic.mpiplot_prepare(tmp.sub(0))
    prep_U1 = plotmagic.mpiplot_prepare(tmp.sub(1))

    tmp = fields[type(solver)]["v"](solver)
    prep_V0 = plotmagic.mpiplot_prepare(tmp.sub(0))
    prep_V1 = plotmagic.mpiplot_prepare(tmp.sub(1))

    tmp = fields[type(solver)]["σ"](solver)
    prep_Q0 = plotmagic.mpiplot_prepare(tmp.sub(0))
    prep_Q1 = plotmagic.mpiplot_prepare(tmp.sub(1))
    prep_Q2 = plotmagic.mpiplot_prepare(tmp.sub(2))
    prep_Q3 = plotmagic.mpiplot_prepare(tmp.sub(3))

    QdG0 = TensorFunctionSpace(mesh, "DG", 0)
    tmp = Function(QdG0)
    prep_QdG0_0 = plotmagic.mpiplot_prepare(tmp.sub(0))
    prep_QdG0_1 = plotmagic.mpiplot_prepare(tmp.sub(1))
    prep_QdG0_2 = plotmagic.mpiplot_prepare(tmp.sub(2))
    prep_QdG0_3 = plotmagic.mpiplot_prepare(tmp.sub(3))

    tmp = Function(solver.P)
    prep_P0 = plotmagic.mpiplot_prepare(tmp.sub(0))
    prep_P1 = plotmagic.mpiplot_prepare(tmp.sub(1))
    prep_P2 = plotmagic.mpiplot_prepare(tmp.sub(2))
    prep_P3 = plotmagic.mpiplot_prepare(tmp.sub(3))

    tmp = Function(Vscalar)
    prep_Vscalar = plotmagic.mpiplot_prepare(tmp)

    QdG0scalar = QdG0.sub(0).collapse()
    tmp = Function(QdG0scalar)
    prep_QdG0scalar = plotmagic.mpiplot_prepare(tmp)

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

    u_ = fields[type(solver)]["u"](solver)
    v_ = fields[type(solver)]["v"](solver)
    σ_ = fields[type(solver)]["σ"](solver)

    def symmetric_vrange(p):
        ignored_minp, maxp = common.minmax(p, take_abs=True, mode="raw")
        return -maxp, maxp

    # remove old colorbars, since `ax.cla` doesn't
    if my_rank == 0:
        print("DEBUG: remove old colorbars")
        for cb in Popper(colorbars):
            cb.remove()

    # u1
    if my_rank == 0:
        print("DEBUG: plot u1")
        ax = axs[0, 0]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(u_.sub(0))
    theplot = plotmagic.mpiplot(u_.sub(0), prep=prep_U0, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$u_{1}$ [m]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # u2
    if my_rank == 0:
        print("DEBUG: plot u2")
        ax = axs[0, 1]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(u_.sub(1))
    theplot = plotmagic.mpiplot(u_.sub(1), prep=prep_U1, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$u_{2}$ [m]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # ε11
    ε_ = solver.εu_
    if my_rank == 0:
        print("DEBUG: plot ε11")
        ax = axs[0, 2]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(0))
    theplot = plotmagic.mpiplot(ε_.sub(0), prep=prep_P0, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\varepsilon_{11}$")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # ε12
    if my_rank == 0:
        print("DEBUG: plot ε12")
        ax = axs[0, 3]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(1))
    theplot = plotmagic.mpiplot(ε_.sub(1), prep=prep_P1, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\varepsilon_{12}$")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # # ε21 - same as ε12 (if the solver works correctly)
    # if my_rank == 0:
    #     ax = axs[XXX, XXX]
    #     ax.cla()
    # vmin, vmax = symmetric_vrange(σ_.sub(2))
    # theplot = plotmagic.mpiplot(σ_.sub(2), prep=prep_P2, show_mesh=show_mesh,
    #                             cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # if my_rank == 0:
    #     colorbars.append(fig.colorbar(theplot, ax=ax))
    #     ax.set_title(r"$\varepsilon_{21}$")
    #     ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    #     ax.set_aspect("equal")

    # ε22
    if my_rank == 0:
        print("DEBUG: plot ε22")
        ax = axs[0, 4]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(3))
    theplot = plotmagic.mpiplot(ε_.sub(3), prep=prep_P3, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\varepsilon_{22}$")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # v1
    if my_rank == 0:
        print(f"DEBUG: plot v1 ≡ {dtext}u1/{dtext}t")
        ax = axs[1, 0]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(v_.sub(0))
    theplot = plotmagic.mpiplot(v_.sub(0), prep=prep_V0, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(f"$v_{{1}} \\equiv {dlatex} u_{{1}} / {dlatex} t$ [m/s]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # v2
    if my_rank == 0:
        print(f"DEBUG: plot v2 ≡ {dtext}u2/{dtext}t")
        ax = axs[1, 1]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(v_.sub(1))
    theplot = plotmagic.mpiplot(v_.sub(1), prep=prep_V1, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(f"$v_{{2}} \\equiv {dlatex} u_{{2}} / {dlatex} t$ [m/s]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # ∂ε11/∂t
    ε_ = solver.εv_
    if my_rank == 0:
        print(f"DEBUG: plot {dtext}ε11/{dtext}t")
        ax = axs[1, 2]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(0))
    theplot = plotmagic.mpiplot(ε_.sub(0), prep=prep_P0, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(f"${dlatex} \\varepsilon_{{11}} / {dlatex} t$ [1/s]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # ∂ε12/∂t
    if my_rank == 0:
        print(f"DEBUG: plot {dtext}ε12/{dtext}t")
        ax = axs[1, 3]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(1))
    theplot = plotmagic.mpiplot(ε_.sub(1), prep=prep_P1, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(f"${dlatex} \\varepsilon_{{12}} / {dlatex} t$ [1/s]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # # ∂ε21/∂t - same as ∂ε12/∂t (if the solver works correctly)
    # if my_rank == 0:
    #     ax = axs[XXX, XXX]
    #     ax.cla()
    #     plt.sca(ax)  # for `plotmagic.mpiplot`
    # vmin, vmax = symmetric_vrange(σ_.sub(2))
    # theplot = plotmagic.mpiplot(σ_.sub(2), prep=prep_P2, show_mesh=show_mesh,
    #                             cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # if my_rank == 0:
    #     colorbars.append(fig.colorbar(theplot, ax=ax))
    #     ax.set_title(f"${dlatex} \\varepsilon_{{21}} / {dlatex} t$ [1/s]")
    #     ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    #     ax.set_aspect("equal")

    # ∂ε22/∂t
    if my_rank == 0:
        print(f"DEBUG: plot {dtext}ε22/{dtext}t")
        ax = axs[1, 4]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(ε_.sub(3))
    theplot = plotmagic.mpiplot(ε_.sub(3), prep=prep_P3, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(f"${dlatex} \\varepsilon_{{22}} / {dlatex} t$ [1/s]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # σ11
    if my_rank == 0:
        print("DEBUG: plot σ11")
        ax = axs[2, 2]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(σ_.sub(0))
    theplot = plotmagic.mpiplot(σ_.sub(0), prep=prep_Q0, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\sigma_{11}$ [Pa]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # σ12
    if my_rank == 0:
        print("DEBUG: plot σ12")
        ax = axs[2, 3]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(σ_.sub(1))
    theplot = plotmagic.mpiplot(σ_.sub(1), prep=prep_Q1, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\sigma_{12}$ [Pa]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # # σ21 - same as σ12 (if the solver works correctly)
    # if my_rank == 0:
    #     ax = axs[XXX, XXX]
    #     ax.cla()
    #     plt.sca(ax)  # for `plotmagic.mpiplot`
    # vmin, vmax = symmetric_vrange(σ_.sub(2))
    # theplot = plotmagic.mpiplot(σ_.sub(2), prep=prep_Q2, show_mesh=show_mesh,
    #                             cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # if my_rank == 0:
    #     colorbars.append(fig.colorbar(theplot, ax=ax))
    #     ax.set_title(r"$\sigma_{21}$ [Pa]")
    #     ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    #     ax.set_aspect("equal")

    # σ22
    if my_rank == 0:
        print("DEBUG: plot σ22")
        ax = axs[2, 4]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    vmin, vmax = symmetric_vrange(σ_.sub(3))
    theplot = plotmagic.mpiplot(σ_.sub(3), prep=prep_Q3, show_mesh=show_mesh,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$\sigma_{22}$ [Pa]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    # We have 13 plots, but 15 subplot slots, so let's use the last two to plot the energy.
    E = project((1 / 2) * inner(σ_, ε(u_)), QdG0scalar)  # elastic strain energy
    if my_rank == 0:
        print("DEBUG: plot elastic strain energy")
        ax = axs[2, 0]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    theplot = plotmagic.mpiplot(E, prep=prep_QdG0scalar, show_mesh=show_mesh)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$(1/2) \sigma : \varepsilon$ [J/m³]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

    K = project((1 / 2) * solver._ρ * dot(v_, v_), Vscalar)  # kinetic energy
    if my_rank == 0:
        print("DEBUG: plot kinetic energy")
        ax = axs[2, 1]
        ax.cla()
        plt.sca(ax)  # for `plotmagic.mpiplot`
    theplot = plotmagic.mpiplot(K, prep=prep_Vscalar, show_mesh=show_mesh)
    if my_rank == 0:
        print("DEBUG: colorbar")
        colorbars.append(fig.colorbar(theplot, ax=ax))
        ax.set_title(r"$(1/2) \rho v^2$ [J/m³]")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal")

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


def export_fields(u_, v_, σ_, *, t):
    """Export solution fields to `.xdmf`."""

    if highres_export_V:
        # Save the displacement visualization at full nodal resolution.
        u_.vector().gather(v_vec_copy, all_V_dofs)  # allgather `u_` to `v_vec_copy`
        v_P1.vector()[:] = v_vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(v_P1, t)

        # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
        v_.vector().gather(v_vec_copy, all_V_dofs)  # allgather `v_` to `v_vec_copy`
        v_P1.vector()[:] = v_vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_v.write(v_P1, t)
    else:  # save at P1 resolution
        xdmffile_u.write(u_, t)
        xdmffile_v.write(v_, t)

    if highres_export_Q:
        σ_.vector().gather(q_vec_copy, all_Q_dofs)
        q_P1.vector()[:] = q_vec_copy[my_Q_dofs]
        xdmffile_σ.write(q_P1, t)
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

        This is the true traceless part, taking the dimensionality
        of `T` as-is.
        """
        d = T.geometric_dimension()
        return T - (1 / d) * tr(T) * Identity(d)
    s = dev(σ_)
    d = s.geometric_dimension()
    dim_to_scale_factor = {3: sqrt(3 / 2), 2: sqrt(2)}
    scale = dim_to_scale_factor[d]
    vonMises_expr = scale * sqrt(inner(s, s))
    vonMises.assign(project(vonMises_expr, Qscalar))
    xdmffile_vonMises.write(vonMises, t)

# --------------------------------------------------------------------------------
# Compute steady-state solution

if not dynamic:
    if my_rank == 0:
        print("Solving steady state...")
    with timer() as solve_tim:
        krylov_it = solver.solve()
    E = elastic_strain_energy()
    if my_rank == 0:  # DEBUG
        print(f"Krylov {krylov_it}; E = ∫ (1/2) σ:ε dΩ = {E:0.3g}; solve wall time {solve_tim.dt:0.3g}s")

    if my_rank == 0:
        print("Saving...")
    u_ = fields[type(solver)]["u"](solver)
    v_ = fields[type(solver)]["v"](solver)
    σ_ = fields[type(solver)]["σ"](solver)
    export_fields(u_, v_, σ_, t=0.0)
    timeseries_u.store(u_.vector(), 0.0)  # the timeseries saves the original data
    timeseries_v.store(v_.vector(), 0.0)
    timeseries_σ.store(σ_.vector(), 0.0)

    if my_rank == 0:
        print("Plotting...")
    msg = "Plotting..."
    with timer() as plot_tim:
        plotit()

    minu, maxu = common.minmax(u_, mode="l2")

    last_plot_walltime_local = plot_tim.dt
    last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
    last_plot_walltime = max(last_plot_walltime_global)
    msg = f"{SUPG_str}V₀ = {V0} m/s; τ = {tau:0.3g} s; |u| ∈ [{minu:0.6g}, {maxu:0.6g}]; solved in {solve_tim.dt:0.2g} s; plot {last_plot_walltime:0.2g} s"
    # Draw one more time to update title.
    if my_rank == 0:
        fig.suptitle(msg)
        plt.tight_layout()
        plotmagic.pause(0.001)
        plt.savefig(f"{fig_output_dir}{fig_basename}_steadystate.{fig_format}")
        plt.ioff()
        print("All done, showing figure.")
        plt.show()
        print("Solver exiting, have a nice day.")
    from sys import exit
    exit(0)

# --------------------------------------------------------------------------------
# Compute dynamic solution

assert dynamic

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

    # Update value in time-dependent boundary conditions
    for expr in (u0_left, u0_right):
        if expr:
            expr.u0 = u0_func(t)

    # Solve one timestep
    # krylov_it1, krylov_it2, krylov_it3, (system_it, last_diff_H1) = solver.step()
    # substeps = 1  # for message
    krylov_it1, krylov_it2, krylov_it3, (system_it, last_diff_H1), (substeps, subdt) = step_adaptive(solver)

    u_ = fields[type(solver)]["u"](solver)
    v_ = fields[type(solver)]["v"](solver)
    σ_ = fields[type(solver)]["σ"](solver)

    if n % nsavemod == 0 or n == nt - 1:
        begin("Saving")
        export_fields(u_, v_, σ_, t=t)
        timeseries_u.store(u_.vector(), t)  # the timeseries saves the original data
        timeseries_v.store(v_.vector(), t)
        timeseries_σ.store(σ_.vector(), t)
        end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # Plot the components of u
    visualize = n % nvismod == 0 or n == nt - 1
    if visualize:
        begin("Plotting")
        with timer() as tim:
            plotit()

            # info for msg (expensive; only update these once per vis step)

            # # On quad elements:
            # #  - `dolfin.interpolate` doesn't work (point/cell intersection only implemented for simplices),
            # #  - `dolfin.project` doesn't work for `dolfin.Expression`, either; same reason.
            # magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=V.ufl_element().degree(),
            #                        u0=u_.sub(0), u1=u_.sub(1))
            # magu = interpolate(magu_expr, Vscalar)
            # uvec = np.array(magu.vector())
            #
            # minu_local = uvec.min()
            # minu_global = MPI.comm_world.allgather(minu_local)
            # minu = min(minu_global)
            #
            # maxu_local = uvec.max()
            # maxu_global = MPI.comm_world.allgather(maxu_local)
            # maxu = max(maxu_global)

            # So let's do this manually. We can operate on the nodal values directly.
            minu, maxu = common.minmax(u_, mode="l2")
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
        print(f"Timestep {n + 1}/{nt}: Krylov {krylov_it1}, {krylov_it2}, {krylov_it3}; system {system_it}; ‖v - v_prev‖_H1 = {last_diff_H1:0.6g}; E = ∫ (1/2) σ:ε dΩ = {E:0.3g}; K = ∫ (1/2) ρ v² dΩ = {K:0.3g}; K + E = {K + E:0.3g}; wall time per timestep {dt_avg:0.3g}s; avg {1/dt_avg:0.3g} timesteps/sec (running avg, n = {len(est.que)})")
        if substeps > 1:
            print(f"    step_adaptive took {substeps} substeps at dt={subdt:0.6g} s")

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
    msg = f"{SUPG_str}t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); V₀ = {V0} m/s; τ = {tau:0.3g} s; |u| ∈ [{minu:0.6g}, {maxu:0.6g}] m; {system_it} iterations; vis every {roundsig(max_vis_step_walltime, 2):g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

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
