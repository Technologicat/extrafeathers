# -*- coding: utf-8; -*-
"""Main program for the coupled problem demo.

TODO: not actually a coupled problem yet.
"""

from enum import IntEnum
import typing

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Function, Expression, Constant, Point,
                    MeshFunction, SubMesh, Facet,
                    near,
                    interpolate,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)
from mshr import Rectangle, Circle, generate_mesh

# custom utilities for FEniCS
from extrafeathers import autoboundary
from extrafeathers import meshutil
from extrafeathers import plotutil

from .navier_stokes import LaminarFlow

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

# mesh_resolution is only used for internal mesh generation
mesh_resolution = 128
nt = 2500

dt = T / nt

# This script expects to be run from the top level of the project as
#   python -m demo.navier_stokes
# or
#   mpirun python -m demo.navier_stokes
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/navier_stokes/flow_over_cylinder_fluid.h5"  # both input and output

# For visualization in ParaView
vis_u_filename = "demo/navier_stokes/velocity.xdmf"
vis_p_filename = "demo/navier_stokes/pressure.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/navier_stokes/velocity_series"
sol_p_filename = "demo/navier_stokes/pressure_series"

# --------------------------------------------------------------------------------
# Mesh generation

# This script supports two ways to generate the mesh:
#  - Run this script itself in serial (non-MPI) mode.
#    A mesh with a uniform cell size will be generated.
#  - Import a Gmsh mesh, by running the `import_gmsh` demo.
#    A graded mesh for this problem is supplied with the demo.

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

# Define boundaries (implicit CompiledSubdomain when fed to DirichletBC)
# inflow = f'near(x[0], {xmin})'
# outflow = f'near(x[0], {xmax})'
# walls = f'near(x[1], {ymin}) || near(x[1], {ymax})'
# cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# `extrafeathers` provides another way to define boundaries: the `autoboundary` module.
#   - Boundary facets between subdomains are extracted automatically.
#   - External boundary facets can be tagged using a callback.
# This produces one MeshFunction (on submesh facets) that has tags for all boundaries.

# Mesh generation
#
# As of FEniCS 2019, marked subdomains are not supported in MPI mode,
# so any mesh generation using them must be done in serial mode.
# https://fenicsproject.discourse.group/t/mpi-and-subdomains/339/2
#
if mpi_comm.size == 1:
    print("Running in serial mode. Generating mesh from hardcoded geometry definition...")

    # Create geometry
    #
    # Note we do not cut out the cylinder; we just mark it. This can be used
    # to generate a subdomain boundary, which can then be tagged with `autoboundary`.
    #
    # We at first mark all cells as belonging to the first subdomain,
    # and then mark the other subdomains (we have only one here).
    domain = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    cylinder = Circle(Point(xcyl, ycyl), rcyl)
    domain.set_subdomain(Domains.FLUID.value, domain)
    domain.set_subdomain(Domains.STRUCTURE.value, cylinder)

    mesh = generate_mesh(domain, mesh_resolution)

    # Create mesh function on cells identifying the marked subdomains.
    # This allows us to extract different subdomains as separate submeshes.
    domain_parts = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
    fluid_mesh = SubMesh(mesh, domain_parts, Domains.FLUID.value)
    structure_mesh = SubMesh(mesh, domain_parts, Domains.STRUCTURE.value)

    # Using the submeshes, we can use `autoboundary` to extract and tag the subdomain boundaries.

    # Specify pairs of subdomains that indicate a physically meaningful subdomain boundary.
    autoboundary_spec = {frozenset({Domains.FLUID.value, Domains.STRUCTURE.value}): Boundaries.OBSTACLE.value}

    # `autoboundary` calls the callback for each facet on the external boundary
    # (i.e. the facet is on a boundary, and no neighboring subdomain exists).
    #
    # For a box domain we could do this, too:
    # xmin, ymin, zmin = domain.first_corner().array()
    # xmax, ymax, zmax = domain.second_corne().array()  # "corne" as of FEniCS 2019.
    def autoboundary_callback(submesh_facet: Facet, fullmesh_facet: Facet) -> typing.Optional[int]:
        p = submesh_facet.midpoint()
        x, y = p.x(), p.y()
        if near(x, xmin):
            return Boundaries.INFLOW.value
        elif near(x, xmax):
            return Boundaries.OUTFLOW.value
        elif near(y, ymin) or near(y, ymax):
            return Boundaries.WALLS.value
        return None  # this facet is not on a boundary we are interested in

    # Tag the boundaries.
    fluid_boundary_parts: MeshFunction = autoboundary.find_subdomain_boundaries(fullmesh=mesh,
                                                                                submesh=fluid_mesh,
                                                                                subdomains=domain_parts,
                                                                                boundary_spec=autoboundary_spec,
                                                                                callback=autoboundary_callback)

    # Save meshes, subdomains and boundary data as HDF5
    #
    # Note, however, that our `domain_parts` are specified w.r.t. `mesh`, not `fluid_mesh`,
    # so they are not applicable here. If we wanted, we could `autoboundary.specialize_meshfunction` it
    # onto `fluid_mesh`. But we don't need the subdomain data in this solver, so we can just leave it out.
    meshutil.write_hdf5_mesh(mesh_filename, fluid_mesh, None, fluid_boundary_parts)

    print("Mesh generated, visualizing.")
    print("Please restart in parallel to solve the problem (mpirun ...)")
    from fenics import plot
    plot(fluid_mesh)
    plot(structure_mesh, color="tan")  # note: not saved to file
    plotutil.plot_facet_meshfunction(fluid_boundary_parts, names=Boundaries)
    plt.legend(loc="best")
    plt.title("Generated mesh")
    plt.show()

    from sys import exit
    exit(0)

# --------------------------------------------------------------------------------
# Solver

# TODO: add command-line argument (using `argparse`) for mesh-generation/solving mode instead of abusing MPI group size
assert mpi_comm.size > 1, "This solver should be run in parallel (mpirun ...)"

if my_rank == 0:
    print("Running in parallel mode. Solving...")

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

xdmffile_u = XDMFFile(mpi_comm, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_p = XDMFFile(mpi_comm, vis_p_filename)
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
