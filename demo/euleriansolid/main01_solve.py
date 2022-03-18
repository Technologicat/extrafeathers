# -*- coding: utf-8; -*-
"""TODO: document this"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (VectorFunctionSpace, TensorFunctionSpace, DirichletBC,
                    Expression, Constant,
                    interpolate, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import EulerianSolid
from .config import (rho, lamda, mu, V0, dt, nt,
                     Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_v_filename, sol_v_filename,
                     vis_σ_filename, sol_σ_filename)

my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = TensorFunctionSpace(mesh, 'P', 1)  # stress, must be one degree lower than `V`

if my_rank == 0:
    print(f"Number of DOFs: displacement {V.dim()}, velocity {V.dim()}, stress {Q.dim()}, total {2 * V.dim() + Q.dim()}")

bcu = []
bcσ = []
solver = EulerianSolid(V, Q, rho, lamda, mu, V0, bcu, bcσ, dt)

# Define boundary conditions
#
bcu_left = DirichletBC(solver.S.sub(0), Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
bcu_right = DirichletBC(solver.S.sub(0), Constant((1, 0)), boundary_parts, Boundaries.RIGHT.value)
bcv_left = DirichletBC(solver.S.sub(1), Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
bcv_right = DirichletBC(solver.S.sub(1), Constant((0, 0)), boundary_parts, Boundaries.RIGHT.value)
# bcσ_right = DirichletBC(solver.S.sub(2), Constant(((1, 0),
#                                                    (0, 0))),
#                         boundary_parts, Boundaries.RIGHT.value)
bcu.append(bcu_left)
bcu.append(bcu_right)
bcu.append(bcv_left)
bcu.append(bcv_right)
# bcσ.append(bcσ_right)

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

# Create time series (for use in other FEniCS solvers)
#
timeseries_u = TimeSeries(sol_u_filename)
timeseries_v = TimeSeries(sol_v_filename)
timeseries_σ = TimeSeries(sol_σ_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# HACK: Arrange things to allow exporting the velocity field at full nodal resolution.
if V.ufl_element().degree() > 1:
    if my_rank == 0:
        print("Preparing export of higher-degree data as refined P1...")
    with timer() as tim:
        func_P1, my_V_dofs = meshmagic.prepare_export_as_P1(V)
        all_V_dofs = np.array(range(V.dim()), "intc")
        vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Enable stabilizers for the Galerkin formulation
solver.stabilizers.SUPG = False  # stabilizer for advection-dominant problems

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
SUPG_str = "[SUPG] " if solver.stabilizers.SUPG else ""  # for messages
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    solver.step()

    begin("Saving")

    if V.ufl_element().degree() > 1:
        # Save the velocity visualization at full nodal resolution.
        solver.u_.vector().gather(vec_copy, all_V_dofs)  # allgather `u_` to `vec_copy`
        func_P1.vector()[:] = vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(func_P1, t)

        # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
        solver.v_.vector().gather(vec_copy, all_V_dofs)  # allgather `v_` to `vec_copy`
        func_P1.vector()[:] = vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_v.write(func_P1, t)
    else:  # save at P1 resolution
        xdmffile_u.write(solver.u_, t)
        xdmffile_v.write(solver.v_, t)
    xdmffile_σ.write(solver.p_, t)
    timeseries_u.store(solver.u_.vector(), t)  # the timeseries saves the original data
    timeseries_v.store(solver.v_.vector(), t)
    timeseries_σ.store(solver.σ_.vector(), t)
    end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # Plot the components of u
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            if my_rank == 0:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 1, 1)
            theplot = plotmagic.mpiplot(solver.u_.sub(0))
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$u_{1}$")
                plt.title(msg)
                plt.subplot(2, 1, 2)
            theplot = plotmagic.mpiplot(solver.u_.sub(1))
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$u_{2}$")

            # info for msg (expensive; only update these once per vis step)
            magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=V.ufl_element().degree(),
                                   u0=solver.u_.sub(0), u1=solver.u_.sub(1))
            magu = interpolate(magu_expr, V.sub(0).collapse())
            uvec = np.array(magu.vector())

            minu_local = uvec.min()
            minu_global = MPI.comm_world.allgather(minu_local)
            minu = min(minu_global)

            maxu_local = uvec.max()
            maxu_global = MPI.comm_world.allgather(maxu_local)
            maxu = max(maxu_global)

            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                plotmagic.pause(0.2)
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
    msg = f"{SUPG_str}; t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.6g}, {maxu:0.6g}]; vis every {max_vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
