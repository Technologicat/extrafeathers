# -*- coding: utf-8; -*-
"""Second pass main program for the coupled problem demo.

Compute the temperature, using the flow field from the first pass for advection.
"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (FunctionSpace, DirichletBC,
                    Expression, Constant,
                    interpolate, project, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end,
                    parameters)

# custom utilities for FEniCS
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import HeatEquation
from .config import (rho, c, k, dt, nt,
                     Boundaries, L,
                     mesh_filename,
                     vis_T_filename, sol_T_filename,
                     sol_u_filename,
                     fig_output_dir, fig_basename, fig_format)

my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function space
V = FunctionSpace(mesh, 'P', 2)  # can use 1 or 2

if my_rank == 0:
    print(f"Number of DOFs: temperature {V.dim()}")

# Define boundary conditions
bc_inflow = DirichletBC(V, Constant(0), boundary_parts, Boundaries.INFLOW.value)
bc_cylinder = DirichletBC(V, Constant(1), boundary_parts, Boundaries.OBSTACLE.value)
bc = [bc_inflow, bc_cylinder]

parameters['krylov_solver']['nonzero_initial_guess'] = True
# parameters['krylov_solver']['monitor_convergence'] = True

# Create XDMF file (for visualization in ParaView)
xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_T.parameters["flush_output"] = True
xdmffile_T.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_T = TimeSeries(sol_T_filename)

# In MPI mode, reading back a saved `TimeSeries` is a bit tricky.
#
# - If we try to read the saved velocity data directly into `solver.a.vector()`, then at the
#   second timestep, PETSc `VecAXPY` fails (when reading the data), saying the arguments are
#   incompatible.
# - If we make a new parallel vector here to hold the data, as `Vector(MPI.comm_world)`,
#   read the data into it, and try to assign that data to `solver.a.vector()[:]`, the
#   assignment fails, because the parallel layouts of the two vectors are not the same.
# - If create a new parallel vector using the copy constructor, `Vector(solver.a.vector())`
#   (so as to copy its parallel layout), read the data into it, and try to assign it to
#   `solver.a.vector()[:]`, we again get the first error above (arguments are incompatible).
#
# To work around this, we read the complete DOF vector (at given time `t`) in all processes,
# and then manually extract the DOFs each process needs. Then the assignment works.
# Thus we make this `TimeSeries` MPI-local (`MPI.comm_self`).
timeseries_velocity = TimeSeries(MPI.comm_self, sol_u_filename)
velocity_alldofs = Vector(MPI.comm_self)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
velocity_degree = 2  # must match stored velocity data
solver = HeatEquation(V, rho, c, k, bc, dt,
                      advection="divergence-free",
                      velocity_degree=velocity_degree)

# Heat source
# h: Function = interpolate(Constant(1.0), V)
# solver.f.assign(f)

# HACK: Arrange things to allow visualizing the temperature field at full nodal resolution.
if V.ufl_element().degree() > 1:
    if my_rank == 0:
        print("Preparing export of higher-degree data as refined P1...")
    with timer() as tim:
        u_P1, my_V_dofs = meshmagic.prepare_linear_export(V)
        all_V_dofs = np.array(range(V.dim()), "intc")
        u_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Analyze mesh and dofmap for plotting (static mesh, only need to do this once)
prep_V = plotmagic.mpiplot_prepare(solver.u_)

# Enable stabilizers for the Galerkin formulation
#
# When Co >> 1 at dense parts of the mesh, convergence may actually be better with SUPG off.
solver.stabilizers.SUPG = True  # stabilizer for advection-dominant problems

advection_velocity = solver.a
my_advection_velocity_dofs = advection_velocity.function_space().dofmap().dofs()

# Time-stepping
t = 0
vis_count = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
SUPG_str = "[SUPG] " if solver.stabilizers.SUPG else ""  # for messages
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Use the velocity field provided by the flow solver.
    #
    # Note this is the end-of-timestep velocity; right now we don't have an initial velocity field saved.
    #
    # The data came from a `VectorFunctionSpace` on our `mesh`, and `advection_velocity` is also
    # a `VectorFunctionSpace` on `mesh`, with the same element degree. Extract the local DOFs
    # (in the MPI sense) from the complete saved DOF vector.
    #
    # We assume to have the same global DOF numbering as in the saved data. Note that if the MPI
    # group size has changed (or if running serially vs. MPI), the DOFs will be numbered differently.
    # (See `demo.dofnumbering` for a demonstration.)
    begin("Loading velocity")
    timeseries_velocity.retrieve(velocity_alldofs, t)
    advection_velocity.vector()[:] = velocity_alldofs[my_advection_velocity_dofs]
    end()

    # Solve one timestep
    solver.step()

    begin("Saving temperature")

    if V.ufl_element().degree() > 1:
        # HACK: What we want to do:
        #   w.assign(interpolate(solver.u_, W))
        # How we do it in MPI mode (see main01_flow.py for full explanation):
        solver.u_.vector().gather(u_copy, all_V_dofs)
        u_P1.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
        u_P1.rename(solver.u_.name(), "a Function")

        xdmffile_T.write(u_P1, t)
    else:  # save at P1 resolution
        xdmffile_T.write(solver.u_, t)
    timeseries_T.store(solver.u_.vector(), t)  # the timeseries saves the original data
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
            theplot = plotmagic.mpiplot(solver.u_, prep=prep_V, cmap="coolwarm")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$T$ (solved)")
                plt.title(msg)
                plt.subplot(2, 1, 2)
            maga_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=velocity_degree,
                                   u0=solver.a.sub(0), u1=solver.a.sub(1))
            maga = interpolate(maga_expr, V)
            theplot = plotmagic.mpiplot(maga, prep=prep_V, cmap="viridis")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$|u|$ (from file)")

            # info for msg (expensive; only update these once per vis step)
            Tvec = np.array(solver.u_.vector())

            minT_local = Tvec.min()
            minT_global = MPI.comm_world.allgather(minT_local)
            minT = min(minT_global)

            maxT_local = Tvec.max()
            maxT_global = MPI.comm_world.allgather(maxT_local)
            maxT = max(maxT_global)

            # Courant number
            Co_adv = project(maga_expr * Constant(dt) / solver.he, V)
            Co_dif = project(solver.ν * Constant(dt) / solver.he**2, V)
            maxCo_adv_local = np.array(Co_adv.vector()).max()
            maxCo_dif_local = np.array(Co_dif.vector()).max()
            maxCo_local = max(maxCo_adv_local, maxCo_dif_local)
            maxCo_global = MPI.comm_world.allgather(maxCo_local)
            maxCo = max(maxCo_global)

            # compute maximum advection velocity, for Péclet number
            maxa_local = np.array(maga.vector()).max()
            maxa_global = MPI.comm_world.allgather(maxa_local)
            maxa = max(maxa_global)

            Pe = solver.peclet(maxa, L)

            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                plotmagic.pause(0.001)
                plt.savefig(f"{fig_output_dir}{fig_basename}_stage02_{vis_count:06d}.{fig_format}")
                vis_count += 1
        last_plot_walltime_local = tim.dt
        last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
        last_plot_walltime = max(last_plot_walltime_global)

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()
    # TODO: make dt, dt_avg part of the public interface in `unpythonic`
    dt_avg = sum(est.que) / len(est.que)
    vis_step_walltime_local = 50 * dt_avg

    # In MPI mode, one of the worker processes may have a larger slice of the domain
    # (or require more Krylov iterations to converge) than the root process.
    # So to get a reliable ETA, we must take the maximum across all processes.
    times_global = MPI.comm_world.allgather((vis_step_walltime_local, est.estimate, est.formatted_eta))
    item_with_max_estimate = max(times_global, key=lambda item: item[1])
    max_eta = item_with_max_estimate[2]
    item_with_max_vis_step_walltime = max(times_global, key=lambda item: item[0])
    max_vis_step_walltime = item_with_max_vis_step_walltime[0]

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{SUPG_str}Pe = {Pe:0.2g}; Co = {maxCo:0.2g}; t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); T ∈ [{minT:0.6g}, {maxT:0.6g}]; vis every {max_vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

# Hold plot
if my_rank == 0:
    print("Simulation complete.")
    plt.ioff()
    plt.show()
