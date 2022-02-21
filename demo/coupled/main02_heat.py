# -*- coding: utf-8; -*-
"""Second pass main program for the coupled problem demo.

Compute the temperature, using the flow field from the first pass for advection.
"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (FunctionSpace, DirichletBC,
                    Expression, Function,
                    interpolate, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshutil
from extrafeathers import plotutil

from .advection_diffusion import AdvectionDiffusion
from .config import (rho, c, k, dt, nt,
                     Boundaries,
                     mesh_filename,
                     vis_T_filename, sol_T_filename,
                     sol_u_filename)
from .util import mypause

my_rank = MPI.rank(MPI.comm_world)

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshutil.read_hdf5_mesh(mesh_filename)

# Define function space
V = FunctionSpace(mesh, 'P', 2)

if my_rank == 0:
    print(f"Number of DOFs: temperature {V.dim()}")

# HACK: Arrange things to allow visualizing the temperature field at full nodal resolution.
if V.ufl_element().degree() == 2:
    if my_rank == 0:
        print("Preparing export of P2 data as refined P1...")
    with timer() as tim:
        export_mesh = plotutil.midpoint_refine(mesh)
        W = FunctionSpace(export_mesh, 'P', 1)
        w = Function(W)
        VtoW, WtoV = plotutil.P2_to_refined_P1(V, W)
        all_V_dofs = np.array(range(V.dim()), "intc")
        u_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
        my_W_dofs = W.dofmap().dofs()  # MPI-local
        my_V_dofs = WtoV[my_W_dofs]  # MPI-local
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Define boundary conditions
bc_inflow = DirichletBC(V, Expression('0', degree=2), boundary_parts, Boundaries.INFLOW.value)
bc_cylinder = DirichletBC(V, Expression('1', degree=2), boundary_parts, Boundaries.OBSTACLE.value)
bc = [bc_inflow, bc_cylinder]

# Create XDMF file (for visualization in ParaView)
xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_T.parameters["flush_output"] = True
xdmffile_T.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_T = TimeSeries(sol_T_filename)

# MPI partitioning may be different in the saved timeseries, so all processes
# must read the complete DOF vector (at given `t`), and then extract the relevant DOFs.
# Thus we make this `TimeSeries` local (`MPI.comm_self`).
timeseries_velocity = TimeSeries(MPI.comm_self, sol_u_filename)
velocity_alldofs = Vector(MPI.comm_self)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
solver = AdvectionDiffusion(V, rho, c, k, bc, dt,
                            advection="divergence-free")

# Heat source
# h: Function = interpolate(Constant(1.0), V)
# solver.f.assign(f)

# Enable stabilizers for the Galerkin formulation
solver.enable_SUPG.b = 0.0  # stabilizer for advection-dominant problems

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
last_plot_walltime_local = 0
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Use the velocity field provided by the flow solver.
    #
    # Note this is the end-of-timestep velocity; right now we don't have an initial velocity field saved.
    #
    # We assume the same mesh and same global DOF numbering as in the saved data.
    # The data came from a `VectorFunctionSpace` on this mesh, and `solver.a` is also
    # a `VectorFunctionSpace` on this mesh. Extract the local DOFs (in the MPI sense)
    # from the complete saved DOF vector.
    begin("Loading velocity")
    timeseries_velocity.retrieve(velocity_alldofs, t)
    advection_velocity = solver.a
    advection_velocity.vector()[:] = velocity_alldofs[advection_velocity.function_space().dofmap().dofs()]
    end()

    # Solve one timestep
    solver.step()

    begin("Saving temperature")

    if V.ufl_element().degree() == 2:
        # HACK: What we want to do:
        #   w.assign(interpolate(solver.u_, W))
        # How we do it in MPI mode (see main01_flow.py for full explanation):
        solver.u_.vector().gather(u_copy, all_V_dofs)
        w.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global

        xdmffile_T.write(w, t)
    else:  # save at P1 resolution
        xdmffile_T.write(solver.u_, t)
    timeseries_T.store(solver.u_.vector(), t)  # the timeseries saves the original P2 data
    end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # Plot T (solved) and the magnitude of u (loaded from file)
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            if my_rank == 0:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
            theplot = plotutil.mpiplot(solver.u_, cmap="coolwarm")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$T$ (solved)")
                plt.title(msg)
                plt.subplot(2, 1, 2)
            magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2,
                                   u0=solver.a.sub(0), u1=solver.a.sub(1))
            magu = interpolate(magu_expr, V)
            theplot = plotutil.mpiplot(magu, cmap="viridis")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$|u|$ (from file)")
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

    Tvec = np.array(solver.u_.vector())

    minT_local = Tvec.min()
    minT_global = MPI.comm_world.allgather(minT_local)
    minT = min(minT_global)

    maxT_local = Tvec.max()
    maxT_global = MPI.comm_world.allgather(maxT_local)
    maxT = max(maxT_global)

    last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
    last_plot_walltime = max(last_plot_walltime_global)

    vis_step_walltime_global = MPI.comm_world.allgather(vis_step_walltime_local)
    vis_step_walltime = max(vis_step_walltime_global)

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); min(T) = {minT:0.6g}; max(T) = {maxT:0.6g}; vis every {vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); wall time {est.formatted_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
