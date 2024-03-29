#!/usr/bin/env python
# -*- coding: utf-8; -*-
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/poisson/demo_poisson.py.rst

from enum import IntEnum
import typing

import numpy as np
from dolfin import (Point, FunctionSpace, Constant, DirichletBC,
                    MeshFunction, TrialFunction, TestFunction, Expression,
                    inner, grad, dx, ds, Facet, FacetNormal,
                    Function, solve, XDMFFile,
                    MPI, parameters, begin, end)
from mshr import Rectangle, generate_mesh

from extrafeathers import autoboundary
from extrafeathers import plotmagic

my_rank = MPI.comm_world.rank

# Create mesh and define function space
N = 16
parameters["ghost_mode"] = "shared_vertex"  # for MPI mode

# Create mesh (L-shaped domain)
bigbox = Rectangle(Point(0, 0), Point(1, 1))
smallbox = Rectangle(Point(0, 0), Point(0.5, 0.5))
domain = bigbox - smallbox
mesh = generate_mesh(domain, N)

V = FunctionSpace(mesh, "P", 2)

# # Define Dirichlet boundary (x = 0 or x = 1)
# def boundary(x):
#     return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define Dirichlet boundary (all external boundaries).
class Boundaries(IntEnum):
    OUTER = 1
# The callback is called for each facet on the outer boundary.
def autoboundary_callback(submesh_facet: Facet, fullmesh_facet: Facet) -> typing.Optional[int]:
    return Boundaries.OUTER.value
boundary_parts: MeshFunction = autoboundary.find_subdomain_boundaries(fullmesh=mesh, submesh=mesh,
                                                                      subdomains=None,
                                                                      boundary_spec={},
                                                                      callback=autoboundary_callback)

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary_parts, Boundaries.OUTER.value)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(mesh)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)

a = inner(grad(u), grad(v)) * dx
L = f * v * dx + g * v * ds

# Compute solution
u = Function(V, name="u")
begin("Solving...")
solve(a == L, u, bc)
begin(f"max(u) = {np.array(u.vector()).max():0.6g}")  # just for MPI-enabled print
end()
end()

# This script expects to be run from the top level of the project as
#   python -m demo.poisson
# or
#   mpirun python -m demo.poisson
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of the path.

# vtkfile = File("demo/output/poisson/solution.pvd")
# vtkfile << u
# Create XDMF file for visualization output
xdmffile_u = XDMFFile(MPI.comm_world, 'demo/output/poisson/u.xdmf')
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.write(u, 0)  # (field, time)

# Visualize
if my_rank == 0:
    print("Plotting.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
theplot = plotmagic.mpiplot(u, show_mesh=True, show_partitioning=True)  # must run in all MPI processes to gather the data
if my_rank == 0:
    plt.colorbar(theplot)
    plt.legend(loc="lower left")  # show the labels for the mesh parts
    plt.axis("equal")
    plt.title(f"Poisson with P{V.ufl_element().degree()} elements")
    plt.show()
