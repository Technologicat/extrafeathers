# -*- coding: utf-8; -*-
"""Investigate the interpolation of different element types on the unit square.

The degree can be specified:

    python -m demo.interptest 1
    python -m demo.interptest 2
    python -m demo.interptest 3

Note this demo exercises also `quad_to_tri` and the quad-plotting functionality
of `extrafeathers`.
"""

import sys

from unpythonic import flatten1

import numpy as np
import matplotlib.pyplot as plt

import dolfin

from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Set up the mesh

N = 2

arg = sys.argv[1] if len(sys.argv) > 1 else "1"
degree = int(arg)

# data = [+1.0, -1.0, -1.0, +1.0]
data = [0.0, 0.0, 0.0, 1.0]
# data = [0.0, 1.0, 1.0, 1.0]
# data = [0.0, 9.0, 4.0, 1.0]

# def bilerp(data, x):
#     """
#     `data`: length-4 vector, layout:
#         2-3
#         | |
#         0-1
#     `x`: length-2 vector, point in unit square
#     """
#     return (data[0] * (1 - x[0]) * (1 - x[1]) +
#             data[1] * x[0] * (1 - x[1]) +
#             data[2] * (1 - x[0]) * x[1] +
#             data[3] * x[0] * x[1])
bilerp = dolfin.Expression("""(d0 * (1.0 - x[0]) * (1.0 - x[1]) +
                              d1 * x[0] * (1.0 - x[1]) +
                              d2 * (1.0 - x[0]) * x[1] +
                              d3 * x[0] * x[1])""",
                           d0=data[0], d1=data[1], d2=data[2], d3=data[3],
                           degree=2)

cases = [(f"{N}×{N} P{degree} (right diagonals)", "P",
          dolfin.UnitSquareMesh(N, N)),
         (f"{N}×{N} P{degree} (crossed diagonals)", "P",
          dolfin.UnitSquareMesh(N, N, "crossed")),
         (f"{N}×{N} Q{degree} (plotted as tri)", "Q",
          dolfin.UnitSquareMesh.create(N, N, dolfin.CellType.Type.quadrilateral))]

if dolfin.MPI.comm_world.rank == 0:
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 8))
    ia = iter(flatten1(ax))
else:
    ia = iter(range(4))

for ((title, family, mesh), a) in zip(cases, ia):
    if dolfin.MPI.comm_world.rank == 0:
        plt.sca(a)
    V = dolfin.FunctionSpace(mesh, family, degree)
    u = dolfin.interpolate(bilerp, V)
    theplot = plotmagic.mpiplot(u)
    if dolfin.MPI.comm_world.rank == 0:
        plt.colorbar(theplot)
        plt.axis("equal")
    plotmagic.mpiplot_mesh(V, show_partitioning=True)
    if dolfin.MPI.comm_world.rank == 0:
        plt.legend(loc="upper right")  # show the labels for the mesh parts
        plt.title(title)

# exact
if dolfin.MPI.comm_world.rank == 0:
    plt.sca(next(ia))
    xx = np.linspace(0, 1, 101)
    X, Y = np.meshgrid(xx, xx, indexing='xy')
    zs = []
    for x, y in zip(np.ravel(X), np.ravel(Y)):
        zs.append(bilerp(x, y))
    Z = np.reshape(zs, np.shape(X))
    theplot = plt.contourf(X, Y, Z, levels=32)
    plt.colorbar(theplot)
    plt.axis("equal")
    plt.title("Exact bilinear")

if dolfin.MPI.comm_world.rank == 0:
    plt.suptitle("Interpolation using different elements")
    plt.show()
