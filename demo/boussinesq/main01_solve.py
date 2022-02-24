# -*- coding: utf-8; -*-
"""Boussinesq flow (natural convection). Two-way coupled problem."""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Expression, Constant, Function,
                    interpolate, project, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import NavierStokes, AdvectionDiffusion
from .config import (rho, mu, c, k, alpha, T0, g, dt, nt,
                     Boundaries, L,
                     mesh_filename,
                     vis_T_filename, sol_T_filename,
                     vis_u_filename, sol_u_filename,
                     vis_p_filename, sol_p_filename)

my_rank = MPI.rank(MPI.comm_world)

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # velocity
Q = FunctionSpace(mesh, 'P', 1)  # pressure
W = FunctionSpace(mesh, 'P', 2)  # temperature

# Set up boundary conditions
bcu_walls = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.WALLS.value)
bcu_obstacle = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.OBSTACLE.value)
bcT_obstacle = DirichletBC(W, Constant(1.0), boundary_parts, Boundaries.OBSTACLE.value)
bcu = [bcu_walls, bcu_obstacle]
bcp = []
bcT = [bcT_obstacle]

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_p = XDMFFile(MPI.comm_world, vis_p_filename)
xdmffile_p.parameters["flush_output"] = True
xdmffile_p.parameters["rewrite_function_mesh"] = False

xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_T.parameters["flush_output"] = True
xdmffile_T.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_p = TimeSeries(sol_p_filename)
timeseries_T = TimeSeries(sol_T_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
#
# TODO: Add support for resuming from the TimeSeries used to store `u_`, `p_`, and `T_`.
#
flowsolver = NavierStokes(V, Q, rho, mu, bcu, bcp, dt)
heatsolver = AdvectionDiffusion(W, rho, c, k, bcT, dt,
                                advection="divergence-free",
                                velocity_degree=2)

# HACK: Arrange things to allow visualizing P2 fields at full nodal resolution.
if V.ufl_element().degree() == 2 or W.ufl_element().degree() == 2:
    if my_rank == 0:
        print("Preparing export of P2 data as refined P1...")
    with timer() as tim:
        if my_rank == 0:
            print("    Mesh...")
        export_mesh = meshmagic.midpoint_refine(mesh)
        if V.ufl_element().degree() == 2:
            if my_rank == 0:
                print("    Velocity...")
            V_export = VectorFunctionSpace(export_mesh, 'P', 1)
            u_export = Function(V_export)
            VtoVexport, VexporttoV = meshmagic.P2_to_refined_P1(V, V_export)
            all_V_dofs = np.array(range(V.dim()), "intc")
            u_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
            my_Vexport_dofs = V_export.dofmap().dofs()  # MPI-local
            my_V_dofs = VexporttoV[my_Vexport_dofs]  # MPI-local
        if W.ufl_element().degree() == 2:
            if my_rank == 0:
                print("    Temperature...")
            W_export = FunctionSpace(export_mesh, 'P', 1)
            T_export = Function(W_export)
            WtoWexport, WexporttoW = meshmagic.P2_to_refined_P1(W, W_export)
            all_W_dofs = np.array(range(W.dim()), "intc")
            T_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on W
            my_Wexport_dofs = W_export.dofmap().dofs()  # MPI-local
            my_W_dofs = WexporttoW[my_Wexport_dofs]  # MPI-local
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Enable stabilizers for the Galerkin formulation
flowsolver.enable_SUPG.b = 1.0  # stabilizer for advection-dominant flows
flowsolver.enable_LSIC.b = 1.0  # additional stabilizer for high Re
heatsolver.enable_SUPG.b = 1.0  # stabilizer for advection-dominant flows

# Boussinesq buoyancy term (as an UFL expression).
#
# This goes into the flow solver as the specific body force.
α = Constant(alpha)
T = heatsolver.u_
ρ_over_ρ0 = 1 - α * (T - T0)  # nondimensional
specific_buoyancy = ρ_over_ρ0 * Constant((0, -g))  # N / kg = m / s²

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
flow_stabilizers_str = "[fS] " if flowsolver.enable_SUPG.b or flowsolver.enable_LSIC.b else ""  # for messages
heat_stabilizers_str = "[hS] " if heatsolver.enable_SUPG.b else ""  # for messages
last_plot_walltime_local = 0
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    begin("Flow solve")
    flowsolver.f.assign(project(specific_buoyancy, V))
    flowsolver.step()
    flowsolver.commit()
    end()
    begin("Heat solve")
    heatsolver.a.assign(flowsolver.u_)
    heatsolver.step()
    heatsolver.commit()
    end()

    begin("Saving")

    if V.ufl_element().degree() == 2:
        flowsolver.u_.vector().gather(u_copy, all_V_dofs)
        u_export.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(u_export, t)
    else:
        xdmffile_u.write(flowsolver.u_, t)
    xdmffile_p.write(flowsolver.p_, t)
    timeseries_u.store(flowsolver.u_.vector(), t)  # the timeseries saves the original P2 data
    timeseries_p.store(flowsolver.p_.vector(), t)

    if W.ufl_element().degree() == 2:
        heatsolver.u_.vector().gather(T_copy, all_W_dofs)
        T_export.vector()[:] = T_copy[my_W_dofs]  # LHS MPI-local; RHS global
        xdmffile_T.write(T_export, t)
    else:
        xdmffile_T.write(heatsolver.u_, t)
    timeseries_T.store(heatsolver.u_.vector(), t)

    end()

    end()

    # Plot p, the magnitude of u, and T
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            # Compute dynamic pressure min/max to center color scale on zero.
            pvec = np.array(flowsolver.p_.vector())

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
                plt.subplot(3, 1, 1)
            theplot = plotmagic.mpiplot(flowsolver.p_, cmap="RdBu_r", vmin=-absmaxp, vmax=+absmaxp)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$p$")
                plt.title(msg)
                plt.subplot(3, 1, 2)
            magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2,
                                   u0=flowsolver.u_.sub(0), u1=flowsolver.u_.sub(1))
            magu = interpolate(magu_expr, V.sub(0).collapse())
            # Courant number of *heat* solver
            Co = project(magu_expr * Constant(dt) / flowsolver.he, V.sub(0).collapse())
            theplot = plotmagic.mpiplot(magu, cmap="viridis")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$|u|$")
                plt.subplot(3, 1, 3)
            theplot = plotmagic.mpiplot(heatsolver.u_, cmap="coolwarm")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$T$")
            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                plotmagic.pause(0.2)
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
    Tvec = np.array(heatsolver.u_.vector())

    minu_local = uvec.min()
    minu_global = MPI.comm_world.allgather(minu_local)
    minu = min(minu_global)

    maxu_local = uvec.max()
    maxu_global = MPI.comm_world.allgather(maxu_local)
    maxu = max(maxu_global)

    minT_local = Tvec.min()
    minT_global = MPI.comm_world.allgather(minT_local)
    minT = min(minT_global)

    maxT_local = Tvec.max()
    maxT_global = MPI.comm_world.allgather(maxT_local)
    maxT = max(maxT_global)

    maxCo_local = np.array(Co.vector()).max()
    maxCo_global = MPI.comm_world.allgather(maxCo_local)
    maxCo = max(maxCo_global)

    last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
    last_plot_walltime = max(last_plot_walltime_global)

    vis_step_walltime_global = MPI.comm_world.allgather(vis_step_walltime_local)
    vis_step_walltime = max(vis_step_walltime_global)

    # Compute the Reynolds and Péclet numbers.
    # We don't have a freestream in this example, so let's use the maximum velocity as representative.
    Re = flowsolver.reynolds(maxu, L)
    Pe = heatsolver.peclet(maxu, L)

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{flow_stabilizers_str}{heat_stabilizers_str}Re = {Re:0.2g}; Pe = {Pe:0.2g}; Co = {maxCo:0.2g}; t = {t + dt:0.2g}; Δt = {dt:0.2g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.2g}, {maxu:0.2g}]; T ∈ [{minT:0.2g}, {maxT:0.2g}]; vis every {vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {est.formatted_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
