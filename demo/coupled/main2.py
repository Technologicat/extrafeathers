# -*- coding: utf-8; -*-
"""Main program for the coupled problem demo.

Compute the temperature in a separate pass.
"""

from enum import IntEnum

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator

from fenics import (FunctionSpace, DirichletBC,
                    Expression,
                    interpolate,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI, Vector,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshutil
from extrafeathers import plotutil

from .advection_diffusion import AdvectionDiffusion

# Matplotlib (3.3.3) has a habit of popping the figure window to top when it is updated using show() or pause(),
# which effectively prevents using the machine for anything else while a simulation is in progress.
#
# To fix this, the suggestion to use the Qt5Agg backend here:
#   https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
#
# didn't help on my system (Linux Mint 20.1). And it is somewhat nontrivial to use a `FuncAnimation` here.
# So we'll use this custom pause function hack instead, courtesy of StackOverflow user @ImportanceOfBeingErnest:
#   https://stackoverflow.com/a/45734500
#
def mypause(interval: float) -> None:
    """Redraw the current figure without stealing focus.

    Works after `plt.show()` has been called at least once.
    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)

mpi_comm = MPI.comm_world
my_rank = MPI.rank(mpi_comm)

# --------------------------------------------------------------------------------
# Settings

rho = 1            # density
c = 1              # specific heat capacity
k = 1e-3           # heat conductivity
T = 5.0            # final time

nt = 2500

dt = T / nt

# This script expects to be run from the top level of the project as
#   python -m demo.navier_stokes
# or
#   mpirun python -m demo.navier_stokes
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/navier_stokes/flow_over_cylinder_fluid.h5"  # both input and output

vis_T_filename = "demo/navier_stokes/temperature.xdmf"
sol_T_filename = "demo/navier_stokes/temperature_series"
sol_u_filename = "demo/navier_stokes/velocity_series"

# --------------------------------------------------------------------------------
# Solver

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshutil.read_hdf5_mesh(mesh_filename)

class Boundaries(IntEnum):  # For Gmsh-imported mesh, these must match the numbering in the .msh file.
    # Autoboundary always tags internal facets with the value 0.
    # Leave it out from the definitions to make the boundary plotter ignore any facet tagged with that value.
    # NOT_ON_BOUNDARY = 0
    INFLOW = 1
    WALLS = 2
    OUTFLOW = 3
    OBSTACLE = 4
class Domains(IntEnum):
    FLUID = 5
    STRUCTURE = 6

# Geometry parameters
xmin, xmax = 0.0, 2.2
half_height = 0.2
xcyl, ycyl, rcyl = 0.2, 0.2, 0.05
ymin = ycyl - half_height
ymax = ycyl + half_height + 0.01  # asymmetry to excite von Karman vortex street

# Define function space
V = FunctionSpace(mesh, 'P', 2)

# Define boundary conditions
# inflow_max = 1.0
# inflow_profile = f'{inflow_max} * 4.0 * (x[1] - {ymin}) * ({ymax} - x[1]) / pow({ymax} - {ymin}, 2)'
# bc_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), boundary_parts, Boundaries.INFLOW.value)
bc_inflow = DirichletBC(V, Expression('0', degree=2), boundary_parts, Boundaries.INFLOW.value)
bc_cylinder = DirichletBC(V, Expression('1', degree=2), boundary_parts, Boundaries.OBSTACLE.value)
bc = [bc_inflow, bc_cylinder]

# Create XDMF files (for visualization in ParaView)
xdmffile_T = XDMFFile(mpi_comm, vis_T_filename)
xdmffile_T.parameters["flush_output"] = True
xdmffile_T.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_T = TimeSeries(sol_T_filename)

# MPI partitioning may be different in the saved timeseries, so all processes
# must read all of each snapshot, and then re-extract the relevant DOFs.
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

# Time-stepping
t = 0
est = ETAEstimator(nt)
for n in range(nt):
    maxT_local = np.array(solver.u_.vector()).max()
    maxT_global = mpi_comm.allgather(maxT_local)
    maxT_str = ", ".join(f"{maxT:0.6g}" for maxT in maxT_global)

    msg = f"{n + 1} / {nt} ({100 * (n + 1) / nt:0.1f}%); t = {t:0.6g}, Δt = {dt:0.6g}; max(T) = {maxT_str}; wall time {est.formatted_eta}"
    begin(msg)

    # Update current time
    t += dt

    # Use the velocity field provided by the flow solver.
    #
    # TODO: fix the one-step offset; right now we don't have an initial velocity field saved.
    # We assume the same mesh and same global DOF numbering as in the saved data.
    # The data came from a `VectorFunctionSpace` on this mesh, and `solver.a` is also
    # a `VectorFunctionSpace` on this mesh. Extract the local DOFs (in the MPI sense).
    begin("Loading velocity")
    timeseries_velocity.retrieve(velocity_alldofs, t)
    advection_velocity = solver.a
    advection_velocity.vector()[:] = velocity_alldofs[advection_velocity.function_space().dofmap().dofs()]
    end()

    # Solve one timestep
    solver.step()

    begin("Saving temperature")
    xdmffile_T.write(solver.u_, t)
    timeseries_T.store(solver.u_.vector(), t)
    end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # Plot T (solved) and the magnitude of u (loaded from file)
    if n % 50 == 0 or n == nt - 1:
        if my_rank == 0:
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 1, 1)
        theplot = plotutil.mpiplot(solver.u_)
        if my_rank == 0:
            plt.axis("equal")
            plt.colorbar(theplot)
            plt.ylabel(r"$T$ (solved)")
            plt.title(msg)
            plt.subplot(2, 1, 2)
        magu = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2,
                          u0=solver.a.sub(0), u1=solver.a.sub(1))
        theplot = plotutil.mpiplot(interpolate(magu, V))
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

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
