# -*- coding: utf-8; -*-
"""First pass main program for the coupled problem demo.

Compute an incompressible flow over a cylinder, for use
as a convection velocity for the temperature field.
"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Expression, Constant,
                    interpolate,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshutil
from extrafeathers import plotutil

from .navier_stokes import LaminarFlow
from .config import (rho, mu, dt, nt,
                     ymin, ymax, Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_p_filename, sol_p_filename)
from .util import mypause

my_rank = MPI.rank(MPI.comm_world)

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshutil.read_hdf5_mesh(mesh_filename)

# TODO: Nondimensionalize properly so that we can use actual physical values of material parameters.
# TODO: Investigate possibilities for a simple FSI solver. Starting point:
#       https://fenicsproject.discourse.group/t/how-can-i-plot-the-boundary-of-a-subdomain/4705/3

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

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
inflow_max = 1.5
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
solver = LaminarFlow(V, Q, rho, mu, bcu, bcp, dt)

# Body force (gravity)
# TODO: vertical gravity requires modification of outlet BCs, because it physically changes the outflow profile.
# f: Function = interpolate(Constant((0, -10.0)), V)
# solver.f.assign(f)

# Time-stepping
t = 0
est = ETAEstimator(nt)
for n in range(nt):
    maxu_local = np.array(solver.u_.vector()).max()
    maxu_global = MPI.comm_world.allgather(maxu_local)
    maxu_str = ", ".join(f"{maxu:0.6g}" for maxu in maxu_global)

    msg = f"{n + 1} / {nt} ({100 * (n + 1) / nt:0.1f}%); t = {t:0.6g}, Δt = {dt:0.6g}; max(u) = {maxu_str}; wall time {est.formatted_eta}"
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    solver.step()

    begin("Saving")
    # TODO: refactor access to u_, p_?
    xdmffile_u.write(solver.u_, t)
    xdmffile_p.write(solver.p_, t)
    timeseries_u.store(solver.u_.vector(), t)
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
        if my_rank == 0:
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 1, 1)
        theplot = plotutil.mpiplot(solver.p_)
        if my_rank == 0:
            plt.axis("equal")
            plt.colorbar(theplot)
            plt.ylabel(r"$p$")
            plt.title(msg)
            plt.subplot(2, 1, 2)
        magu = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2,
                          u0=solver.u_.sub(0), u1=solver.u_.sub(1))
        theplot = plotutil.mpiplot(interpolate(magu, V.sub(0).collapse()))
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

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
