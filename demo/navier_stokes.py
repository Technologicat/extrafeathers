#!/usr/bin/env python
# -*- coding: utf-8; -*-
"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0

Customized version:

  - Marked submeshes for automatic obstacle boundary extraction.
  - Parallel computation using MPI.
  - Visualization for ongoing simulation also in MPI mode.
"""

from enum import IntEnum
import typing

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Expression, Constant, Point,
                    Function, TrialFunction, TestFunction, FacetNormal,
                    MeshFunction, SubMesh, Mesh, Facet,
                    dot, inner, sym,
                    nabla_grad, div, dx, ds,
                    Identity,
                    DOLFIN_EPS,
                    lhs, rhs, assemble, solve, interpolate,
                    HDF5File, XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)
from mshr import Rectangle, Circle, generate_mesh

# custom utilities for FEniCS
from extrafeathers import autoboundary
from extrafeathers import plotutil

mpi_comm = MPI.comm_world
my_rank = MPI.rank(mpi_comm)

# --------------------------------------------------------------------------------
# Settings

mu = 0.001         # dynamic viscosity
rho = 1            # density
T = 5.0            # final time

# mesh_resolution is only used for internal mesh generation
mesh_resolution = 64
nt = 5000

dt = T / nt

mesh_filename = "demo/navier_stokes/flow_over_cylinder.h5"  # both input and output

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

class Domains(IntEnum):
    FLUID = 1
    STRUCTURE = 2
class Boundaries(IntEnum):  # For Gmsh-imported mesh, these must match the numbering in the .msh file.
    # Autoboundary always tags internal facets with the value 0.
    # Leave it out from the definitions to make the boundary plotter ignore any facet tagged with that value.
    # NOT_ON_BOUNDARY = 0
    INFLOW = 1
    WALLS = 2
    OUTFLOW = 3
    OBSTACLE = 4

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
    def near(x1: float, x2: float, tol: float = DOLFIN_EPS) -> bool:
        return abs(x1 - x2) <= tol

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
    boundary_parts: MeshFunction = autoboundary.find_subdomain_boundaries(submesh=fluid_mesh, fullmesh=mesh,
                                                                          subdomains=domain_parts,
                                                                          boundary_spec=autoboundary_spec,
                                                                          callback=autoboundary_callback)

    # Save meshes, subdomains and boundary data as HDF5
    with HDF5File(mesh.mpi_comm(), mesh_filename, "w") as hdf:
        hdf.write(fluid_mesh, "/mesh")
        hdf.write(boundary_parts, "/boundary_parts")  # MeshFunction on facets of `fluid_mesh`

    print("Mesh generated, visualizing.")
    print("Please restart in parallel to solve the problem (mpirun ...)")
    from fenics import plot
    plot(fluid_mesh)
    plot(structure_mesh, color="tan")  # note: not saved to file
    plotutil.plot_facet_meshfunction(boundary_parts, names=Boundaries)
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
with HDF5File(mpi_comm, mesh_filename, "r") as hdf:
    mesh = Mesh()
    hdf.read(mesh, "/mesh", False)
    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
    hdf.read(boundary_parts, "/boundary_parts")

# TODO: Nondimensionalize properly so that we can use actual physical values of material parameters.
# TODO: Investigate possibilities for a simple FSI solver. Starting point:
#       https://fenicsproject.discourse.group/t/how-can-i-plot-the-boundary-of-a-subdomain/4705/3

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')  # parabolic, 1.5 at the middle

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

# Trial and test functions
u = TrialFunction(V)  # no suffix: the UFL symbol for the unknown quantity
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Functions for solution at previous and current time steps
u_n = Function(V)  # suffix _n: the old value (end of previous timestep)
u_ = Function(V)  # suffix _: the latest computed approximation
p_n = Function(Q)
p_ = Function(Q)

# TODO: Initialize u_ and p_ from a potential-flow approximation to have a physically reasonable initial state.
# TODO: We need to implement a potential-flow solver to be able to do that. It's just a Poisson equation,
# TODO: but the scalings must match this solver, and we need to produce a pressure field, too.

# Expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)  # wrap in a Constant to allow changing the value without triggering a recompile
mu = Constant(mu)
rho = Constant(rho)

# Symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Stress tensor (isotropic Newtonian fluid)
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

# Define variational problem for step 1 (tentative velocity)
#
# This is just the variational form of the momentum equation.
#
# The boundary terms must match the implementation of the sigma term. Note we use the
# old pressure p_n and the midpoint value of velocity U.
#
#   U = (1/2) (u_n + u)
# so
#   sigma = mu * epsilon(u_n + u) - p_n * I
# Integrating -div(sigma) * v dx by parts, we get sigma : grad(v) dx = sigma : symm_grad(v) dx
# (we can use symm_grad, because sigma is symmetric) plus the boundary term
#   -sigma*n*v*ds = -mu * epsilon(u_n + u) * n * v * ds  +  p_n * n * v * ds
#
# Then requiring du/dn = 0 on the Neumann boundary (for fully developed outflow) eliminates one
# of the terms from inside the epsilon in the boundary term, leaving just the other one, and the
# pressure term. Here we use the transpose jacobian convention for the gradient of a vector,
# (∇u)ik := ∂i uk, so we must keep the term  (∂ui uk) nk  (this comes from the transposed part).
#
F1 = (rho * dot((u - u_n) / k, v) * dx +
      rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx +
      inner(sigma(U, p_n), epsilon(v)) * dx +
      dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds -
      dot(f, v) * dx)
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2 (pressure correction)
#
# Subtract the momentum equation, written in terms of tentative velocity u_ and old pressure p_n,
# from the momentum equation written in terms of new unknown velocity u and new unknown pressure p.
#
# The momentum equation is
#
#   ρ ( ∂u/∂t + u·∇u ) = ∇·σ + f
#                       = ∇·(μ symm∇u - p I) + f
#                       = ∇·(μ symm∇u) - ∇p + f
# so we have
#
#   ρ ( ∂(u - u_)/∂t + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
#
# Discretizing the time derivative,
#
#   ρ ( (u - u_n)/k - (u_ - u_n)/k + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
#
# Canceling the u_n,
#
#   ρ ( (u - u_)/k + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
#
# Rearranging,
#   ρ (u - u_)/k + ∇(p - p_n) = ∇·(μ symm∇(u - u_)) - ρ (u - u_)·∇(u - u_)
#
# Now, if u_ is "close enough" to u, we may take the RHS to be zero (Goda, 1979;
# see also e.g. Landet and Mortensen, 2019, section 3).
#
# The result is the velocity correction equation, which we will use in step 3 below:
#
#   ρ (u - u_) / k + ∇p - ∇p_n = 0
#
# For step 2, take the divergence of the velocity correction equation, and use the continuity
# equation to eliminate div(u) (it must be zero for the new unknown velocity); obtain a Poisson
# problem for the new pressure p, in terms of the old pressure p_n and the tentative velocity u_:
#
#   -ρ ∇·u_ / k + ∇²p - ∇²p_n = 0
#
# See also Langtangen and Logg (2016, section 3.4).
#
#   Katuhiko Goda. A multistep technique with implicit difference schemes for calculating
#   two- or three-dimensional cavity flows. Journal of Computational Physics, 30(1):76–95,
#   1979.
#
#   Tormod Landet, Mikael Mortensen, 2019. On exactly incompressible DG FEM pressure splitting
#   schemes for the Navier-Stokes equation. arXiv: 1903.11943v1
#
#   Hans Petter Langtangen, Anders Logg (2016). Solving PDEs in Python: The FEniCS Tutorial 1.
#   Simula Springer Briefs on Computing 3.
#
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx

# Define variational problem for step 3 (velocity correction)
a3 = rho * dot(u, v) * dx
L3 = rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble matrices (constant in time; do this once at the start)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files (for visualization in ParaView)
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
timeseries_u = TimeSeries(sol_u_filename)
timeseries_p = TimeSeries(sol_p_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Time-stepping
t = 0
est = ETAEstimator(nt)
for n in range(nt):
    maxu_local = np.array(u_.vector()).max()
    maxu_global = mpi_comm.allgather(maxu_local)
    maxu_str = ", ".join(f"{maxu:0.6g}" for maxu in maxu_global)

    msg = f"{n + 1} / {nt} ({100 * (n + 1) / nt:0.1f}%); t = {t:0.6g}, Δt = {dt:0.6g}; max(u) = {maxu_str}; wall time {est.formatted_eta}"
    begin(msg)

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    begin("Tentative velocity")
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
    end()

    # Step 2: Pressure correction step
    begin("Pressure correction")
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    end()

    # Step 3: Velocity correction step
    begin("Velocity correction")
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')
    end()

    begin("Saving")
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    end()

    # Update progress bar
    progress += 1
    est.tick()
    end()
    # print(f'Process {my_rank}, timestep done: t = {t:0.6g}: max(u) = {np.array(u_.vector()).max():0.6g}')

    # # Plot p and the components of u
    # if n % 50 == 0 or n == nt - 1:
    #     if my_rank == 0:
    #         plt.figure(1)
    #         plt.clf()
    #         plt.subplot(3, 1, 1)
    #     theplot = plotutil.mpiplot(p_)
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$p$")
    #         plt.title(msg)
    #         plt.subplot(3, 1, 2)
    #     theplot = plotutil.mpiplot(u_.sub(0))
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$u_x$")
    #         plt.subplot(3, 1, 3)
    #     theplot = plotutil.mpiplot(u_.sub(1))
    #     if my_rank == 0:
    #         plt.axis("equal")
    #         plt.colorbar(theplot)
    #         plt.ylabel(r"$u_y$")
    #     if my_rank == 0:
    #         plt.draw()
    #         # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
    #         plt.pause(0.5)

    # Plot p and the magnitude of u
    if n % 50 == 0 or n == nt - 1:
        if my_rank == 0:
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 1, 1)
        theplot = plotutil.mpiplot(p_)
        if my_rank == 0:
            plt.axis("equal")
            plt.colorbar(theplot)
            plt.ylabel(r"$p$")
            plt.title(msg)
            plt.subplot(2, 1, 2)
        magu = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=2, u0=u_.sub(0), u1=u_.sub(1))
        theplot = plotutil.mpiplot(interpolate(magu, V.sub(0).collapse()))
        if my_rank == 0:
            plt.axis("equal")
            plt.colorbar(theplot)
            plt.ylabel(r"$|u|$")
        if my_rank == 0:
            plt.draw()
            # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
            plt.pause(0.5)

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
