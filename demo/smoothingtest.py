# -*- coding: utf-8; -*-
"""Demonstrate the effects of Q1->Q0->Q1 projection smoothing."""

import dolfin

import matplotlib.pyplot as plt

from extrafeathers import plotmagic

N = 4

mesh = dolfin.UnitSquareMesh.create(N, N, dolfin.CellType.Type.quadrilateral)
V1 = dolfin.FunctionSpace(mesh, "Q", 1)
V0 = dolfin.FunctionSpace(mesh, "DQ", 0)

# x = dolfin.SpatialCoordinate(mesh)
# ax = N * dolfin.pi * x[0]
# ay = N * dolfin.pi * x[1]
# fh = dolfin.project(dolfin.sin(ax) * dolfin.sin(ay), V1)

f = dolfin.Expression("sin(N * pi * x[0]) * sin(N * pi * x[1])", N=N, degree=3)
fh = dolfin.interpolate(f, V0)

if dolfin.MPI.comm_world.rank == 0:
    fig, ax = plt.subplots(1, 2, constrained_layout=True)

if dolfin.MPI.comm_world.rank == 0:
    plt.sca(ax[0])
theplot = plotmagic.mpiplot(fh, show_mesh=True, show_partitioning=True, vmin=-1.0, vmax=1.0, cmap="Greys")
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)
    plt.axis("equal")
    plt.title("Original DQ0 function")

if dolfin.MPI.comm_world.rank == 0:
    plt.sca(ax[1])
fh = dolfin.project(fh, V1)
theplot = plotmagic.mpiplot(fh, show_mesh=True, show_partitioning=True, vmin=-1.0, vmax=1.0, cmap="Greys")
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)
    plt.axis("equal")
    plt.title("Projected to Q1")

if dolfin.MPI.comm_world.rank == 0:
    plt.show()
