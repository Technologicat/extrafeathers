# -*- coding: utf-8; -*-
"""Valentine's day card using the Navier-Stokes solver."""

from enum import IntEnum

import numpy as np
import matplotlib
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

from demo.coupled.navier_stokes import LaminarFlow

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

mu = 0.001         # dynamic viscosity
rho = 1            # density
T = 5.0            # final time

nt = 2500

dt = T / nt

# This script expects to be run from the top level of the project as
#   python -m demo.navier_stokes
# or
#   mpirun python -m demo.navier_stokes
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

gmsh_mesh_filename = "demo/meshes/heart.msh"  # input
h5_mesh_filename = "demo/meshes/heart.h5"  # output

# For visualization in ParaView
vis_u_filename = "demo/output/valentinesday/velocity.xdmf"
vis_p_filename = "demo/output/valentinesday/pressure.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/valentinesday/velocity_series"
sol_p_filename = "demo/output/valentinesday/pressure_series"

class Boundaries(IntEnum):  # For Gmsh-imported mesh, these must match the numbering in the .msh file.
    # Autoboundary always tags internal facets with the value 0.
    # Leave it out from the definitions to make the boundary plotter ignore any facet tagged with that value.
    # NOT_ON_BOUNDARY = 0
    INFLOW_R = 1
    INFLOW_L = 2
    WALLS = 3
class Domains(IntEnum):
    FLUID = 4

# --------------------------------------------------------------------------------
# Import the mesh

if mpi_comm.size == 1:
    print("Running in serial mode. Importing mesh...")
    meshutil.import_gmsh(gmsh_mesh_filename, h5_mesh_filename)
    from sys import exit
    exit(0)

# --------------------------------------------------------------------------------
# Solver

assert mpi_comm.size > 1, "This solver should be run in parallel (mpirun ...)"

if my_rank == 0:
    print("Running in parallel mode. Solving...")

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshutil.read_hdf5_mesh(h5_mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
xmax = 0.5  # end of inlet, x direction; must not be longer than the part in mesh
umax = 1.0
inflow_profile_r = f"max(0.0, {umax} * 4.0 * x[0] * ({xmax} - x[0]) / pow({xmax}, 2))"
bcu_inflow_r = DirichletBC(V, Expression((f'{inflow_profile_r}', f'{inflow_profile_r}'), degree=2), boundary_parts, Boundaries.INFLOW_R.value)
inflow_profile_l = f"max(0.0, {umax} * 4.0 * -x[0] * ({xmax} + x[0]) / pow({xmax}, 2))"
bcu_inflow_l = DirichletBC(V, Expression((f'-{inflow_profile_l}', f'{inflow_profile_l}'), degree=2), boundary_parts, Boundaries.INFLOW_L.value)
bcu_walls = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.WALLS.value)
bcu = [bcu_inflow_r, bcu_inflow_l, bcu_walls]
bcp = []  # no Dirichlet BCs on pressure; Krylov solvers can handle singular systems just fine.

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(mpi_comm, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_p = XDMFFile(mpi_comm, vis_p_filename)
xdmffile_p.parameters["flush_output"] = True
xdmffile_p.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_p = TimeSeries(sol_p_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
solver = LaminarFlow(V, Q, rho, mu, bcu, bcp, dt)

# Time-stepping
t = 0
est = ETAEstimator(nt)
for n in range(nt):
    maxu_local = np.array(solver.u_.vector()).max()
    maxu_global = mpi_comm.allgather(maxu_local)
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
