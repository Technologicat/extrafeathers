# -*- coding: utf-8; -*-
"""First pass main program for the coupled problem demo.

Compute an incompressible flow over a cylinder, for use
as a convection velocity for the temperature field.
"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Expression, Constant, Function,
                    interpolate, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshiowrapper
from extrafeathers import plotutil

from .navier_stokes import NavierStokes
from .config import (rho, mu, dt, nt, inflow_max,
                     Boundaries, L,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_p_filename, sol_p_filename)
from .util import mypause

my_rank = MPI.rank(MPI.comm_world)

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# TODO: Nondimensionalize properly so that we can use actual physical values of material parameters.
# TODO: Investigate possibilities for a simple FSI solver. Starting point:
#       https://fenicsproject.discourse.group/t/how-can-i-plot-the-boundary-of-a-subdomain/4705/3

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Detect ymin/ymax for configuring inflow profile.
with timer() as tim:
    ignored_cells, nodes_dict = plotutil.all_cells(Q)
    ignored_dofs, nodes_array = plotutil.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])

if my_rank == 0:
    print(f"Geometry detection completed in {tim.dt:0.6g} seconds.")
    print(f"x ∈ [{xmin:0.6g}, {xmax:0.6g}], y ∈ [{ymin:0.6g}, {ymax:0.6g}].")
    print(f"Number of DOFs: velocity {V.dim()}, pressure {Q.dim()}, total {V.dim() + Q.dim()}")

# Define boundary conditions
#
# We want a parabolic inflow profile, with a given maximum value at the middle.
# As SymPy can tell us, the parabola that interpolates from 0 to 1 and back to 0
# in the interval `(a, b)` is the following `f`:
#
# import sympy as sy
# x, a, b = sy.symbols("x, a, b")
# f = 4 * (x - a) * (b - x) / (b - a)**2
# assert sy.simplify(f.subs({x: a})) == 0
# assert sy.simplify(f.subs({x: b})) == 0
# assert sy.simplify(f.subs({x: a + (b - a) / 2})) == 1
#
# Thus, the C++ code for the inflow profile Expression is:
inflow_profile = (f'{inflow_max} * 4.0 * (x[1] - {ymin}) * ({ymax} - x[1]) / pow({ymax} - {ymin}, 2)', '0')

# # using the implicit CompiledSubdomain, as in FEniCS tutorial:
# bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
# bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
# bcp_outflow = DirichletBC(Q, Constant(0), outflow)

# # using tags from mesh file:
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), boundary_parts, Boundaries.INFLOW.value)
bcu_walls = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.WALLS.value)
bcp_outflow = DirichletBC(Q, Constant(0), boundary_parts, Boundaries.OUTFLOW.value)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.OBSTACLE.value)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Create XDMF files (for visualization in ParaView)
#
# As the `h5ls` tool will tell you, these files contain a mesh, and values of the
# function on each that mesh. Vector quantities (such as `u`) are represented in
# cartesian component form.
#
# Setting the "flush_output" flag makes the file viewable even when the simulation is
# still running, terminated early (Ctrl+C), or crashed (no convergence).
#
# In our test case the mesh is constant in time, so there is no point in rewriting it
# for each timestep. This results in a much smaller file size, which is significant,
# since HDF5 compression is not supported in parallel mode. If you need to compress
# simulation results to save disk space, use the `h5repack` tool on the HDF5 files
# after the simulation is complete.
#
# https://fenicsproject.discourse.group/t/how-to-enable-compression-of-the-hdf5-file-associated-with-xdmffile/793
# https://fenicsproject.org/qa/6068/enabling-hdf5-compression/

xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_p = XDMFFile(MPI.comm_world, vis_p_filename)
xdmffile_p.parameters["flush_output"] = True
xdmffile_p.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
#
# These files contain the raw DOF data.
#
timeseries_u = TimeSeries(sol_u_filename)
timeseries_p = TimeSeries(sol_p_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
#
# TODO: Add support for resuming from the TimeSeries used to store `u_` and `p_`.
#
# TODO: Initialize `u_` and `p_` from a potential-flow approximation to have a physically reasonable initial state.
# TODO: We need to implement a potential-flow solver to be able to do that. It's just a Poisson equation,
# TODO: but the scalings must match this solver, and we need to produce a pressure field, too.
#
solver = NavierStokes(V, Q, rho, mu, bcu, bcp, dt)

# Body force (gravity)
# TODO: vertical gravity requires modification of outlet BCs, because it physically changes the outflow profile.
# f: Function = interpolate(Constant((0, -10.0)), V)
# solver.f.assign(f)

# HACK: Arrange things to allow visualizing the velocity field at full nodal resolution.
# TODO: is it possible to export curved (quadratic isoparametric) FEM data to ParaView?
#
# For `u`, we use the P2 element, which does not export at full quality into
# vertex-based formats. The edge midpoint nodes are (obviously) not at the
# vertices of the mesh, so the exporter simply ignores those DOFs.
#
# Thus, if we save the P2 data for visualization directly, the solution will
# appear to have a much lower resolution than it actually does. This is
# especially noticeable at parts of the mesh where the local element size
# is large. This is unfortunate, as halving the element size in 2D requires
# (roughly) 4× more computation time.
#
# So, given that the P2 space already gives us additional DOF data, can we
# improve the export quality at a given element size?
#
# By refining the mesh once, we can obtain a new mesh that *does* have
# additional vertices at the midpoints of the edges of the original mesh.
# We can then set up a P1 function space on this refined mesh. Pretending that
# the data is P1, we can just map the DOF data onto the new mesh (i.e. essentially
# `dolfin.interpolate` it onto the P1 space), export that as P1, and it'll work
# in any vertex-based format.
#
# Obviously, the exported field does not match the computed one exactly (except
# at the nodes), because it has not been assembled from the P2 Galerkin series,
# but instead the coefficients have been recycled as-is for use in a somewhat
# related P1 Galerkin series.
#
# An L2 projection onto the P1 space would be better, but it is costly, and
# importantly, tricky to do in parallel when the meshes have different MPI
# partitioning.
#
# Even the simple interpolation approach gives a marked improvement on the
# visual quality of the exported data, at a small fraction of the cost,
# so that's what we use.
#
# We'll need a P1 FEM function on the refined mesh to host the visualization
# DOFs for saving. It is important to re-use the same `Function` object
# instance when saving each timestep, because by default constructing a
# `Function` gives the field a new name (unless you pass in `name=...`).
# ParaView, on the other hand, needs the name to stay the same over the whole
# simulation to recognize it as the same field. (FEniCS's default is "f_xxx"
# for a running number xxx, incremented each time a `Function` is created.)
#
# Because the original mesh has P2 Lagrange elements, and the refined mesh
# has P1 Lagrange elements, we can interpolate by simply copying the DOF values
# at the coincident nodes. So we just need a mapping for the global DOF vector
# that takes the data from the P2 function space DOFs to the corresponding P1
# function space DOFs.
#
if V.ufl_element().degree() == 2:
    if my_rank == 0:
        print("Preparing export of P2 data as refined P1...")
    with timer() as tim:
        export_mesh = plotutil.midpoint_refine(mesh)
        W = VectorFunctionSpace(export_mesh, 'P', 1)
        w = Function(W)
        VtoW, WtoV = plotutil.P2_to_refined_P1(V, W)
        all_V_dofs = np.array(range(V.dim()), "intc")
        u_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
        my_W_dofs = W.dofmap().dofs()  # MPI-local
        my_V_dofs = WtoV[my_W_dofs]  # MPI-local
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

Re = solver.reynolds(inflow_max, L)

# Enable stabilizers for the Galerkin formulation
solver.enable_SUPG.b = 1.0  # stabilizer for advection-dominant flows
solver.enable_LSIC.b = 1.0  # additional stabilizer for high Re

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
SUPG_str = "[SUPG] " if solver.enable_SUPG.b else ""  # for messages
LSIC_str = "[LSIC] " if solver.enable_LSIC.b else ""  # for messages
last_plot_walltime_local = 0
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    solver.step()

    begin("Saving")

    if V.ufl_element().degree() == 2:
        # Save the velocity visualization at full nodal resolution (we have a P2 space!).
        #
        # HACK: What we want to do:
        #
        #   w.assign(interpolate(solver.u_, W))
        #
        # In MPI mode, the problem is that the DOFs of V and W partition differently,
        # so each MPI process has no access to some of the `solver.u_` data it needs
        # to construct its part of `w`.
        #
        # One option would be to make a separate serial postprocess script
        # that loads `u_` from the timeseries file (on the original P2 space;
        # now not partitioned because serial mode), performs this interpolation,
        # and generates the visualization file.
        #
        # But using the DOF mappings defined above, we can generate the visualization
        # right now, in MPI mode. We allgather the DOFs of the solution on V, and then
        # remap them onto the corresponding DOFS on W:
        solver.u_.vector().gather(u_copy, all_V_dofs)
        w.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
        # Now `w` is a refined P1 representation of the velocity field.

        # TODO: refactor access to u_, p_?
        xdmffile_u.write(w, t)
    else:  # save at P1 resolution
        xdmffile_u.write(solver.u_, t)
    xdmffile_p.write(solver.p_, t)
    timeseries_u.store(solver.u_.vector(), t)  # the timeseries saves the original P2 data
    timeseries_p.store(solver.p_.vector(), t)
    end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # # Plot p and the components of u
    # if n % 50 == 0 or n == nt - 1:
    #     if my_rank == 0:
    #         plt.figure(1)
    #         plt.clf()
    #         plt.subplot(3, 1, 1)
    #     theplot = plotutil.mpiplot(solver.p_)
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$p$")
    #         plt.title(msg)
    #         plt.subplot(3, 1, 2)
    #     theplot = plotutil.mpiplot(solver.u_.sub(0))
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$u_x$")
    #         plt.subplot(3, 1, 3)
    #     theplot = plotutil.mpiplot(solver.u_.sub(1))
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$u_y$")
    #     if my_rank == 0:
    #         plt.draw()
    #         if n == 0:
    #             plt.show()
    #         # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
    #         mypause(0.2)

    # Plot p and the magnitude of u
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            # Compute dynamic pressure min/max to center color scale on zero.
            pvec = np.array(solver.p_.vector())

            minp_local = pvec.min()
            minp_global = MPI.comm_world.allgather(minp_local)
            minp = min(minp_global)

            maxp_local = pvec.max()
            maxp_global = MPI.comm_world.allgather(maxp_local)
            maxp = max(maxp_global)

            absmaxp = max(abs(minp), abs(maxp))

            if my_rank == 0:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
            theplot = plotutil.mpiplot(solver.p_, cmap="RdBu_r", vmin=-absmaxp, vmax=+absmaxp)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$p$")
                plt.title(msg)
                plt.subplot(2, 1, 2)
            magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2,
                                   u0=solver.u_.sub(0), u1=solver.u_.sub(1))
            magu = interpolate(magu_expr, V.sub(0).collapse())
            theplot = plotutil.mpiplot(magu, cmap="viridis")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$|u|$")
            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                mypause(0.2)
        if my_rank == 0:
            last_plot_walltime_local = tim.dt

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()
    # TODO: make dt, dt_avg part of the public interface in `unpythonic`
    dt_avg = sum(est.que) / len(est.que)
    vis_step_walltime_local = 50 * dt_avg

    uvec = np.array(magu.vector())

    minu_local = uvec.min()
    minu_global = MPI.comm_world.allgather(minu_local)
    minu = min(minu_global)

    maxu_local = uvec.max()
    maxu_global = MPI.comm_world.allgather(maxu_local)
    maxu = max(maxu_global)

    last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
    last_plot_walltime = max(last_plot_walltime_global)

    vis_step_walltime_global = MPI.comm_world.allgather(vis_step_walltime_local)
    vis_step_walltime = max(vis_step_walltime_global)

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{SUPG_str}{LSIC_str}Re = {Re:0.2g}; t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.6g}, {maxu:0.6g}]; vis every {vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {est.formatted_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
