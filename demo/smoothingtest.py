# -*- coding: utf-8; -*-
"""Demonstrate the effects of Q1->Q0->Q1 projection smoothing.

Cell kind ("Q" or "P") can be specified on the command line:

    python -m demo.smoothingtest Q
    python -m demo.smoothingtest P
"""

import sys

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import common
from extrafeathers import meshmagic
from extrafeathers import plotmagic


N = 8

family = sys.argv[1] if len(sys.argv) > 1 else "Q"

if family == "Q":
    mesh = dolfin.UnitSquareMesh.create(N, N, dolfin.CellType.Type.quadrilateral)
else:
    mesh = dolfin.UnitSquareMesh(N, N, "crossed")

V1 = dolfin.FunctionSpace(mesh, "P", 1)
V0 = dolfin.FunctionSpace(mesh, "DP", 0)

# x = dolfin.SpatialCoordinate(mesh)
# ax = N * dolfin.pi * x[0]
# ay = N * dolfin.pi * x[1]
# fh = dolfin.project(dolfin.sin(ax) * dolfin.sin(ay), V1)

# Original function
f = dolfin.Expression("x[0] * x[1]", degree=3)
fh = dolfin.interpolate(f, V1)

# Stand-in for a checkerboard numerical oscillation we want to eliminate.
#
# # Checkerboard function (when sampled on DQ0)
# g = dolfin.Expression("1e6 * sin(N * pi * x[0]) * sin(N * pi * x[1])", N=N, degree=3)
# gh = dolfin.interpolate(g, V0)
#
# Checkerboard function at nodes of Q1
g = dolfin.Expression("1e6 * cos(N * pi * x[0]) * cos(N * pi * x[1])", N=N, degree=3)
gh = dolfin.interpolate(g, V1)

def doit(suptitle, error=None):
    # Add simulated checkerboard oscillation
    fh_distorted = dolfin.Function(fh.function_space(), name="fh_distorted")
    if error:
        fh_distorted.vector()[:] = fh.vector()[:] + error.vector()[:]
    else:
        fh_distorted.vector()[:] = fh.vector()[:]

    # # DEBUG
    # theplot = plotmagic.mpiplot(gh)
    # plt.colorbar(theplot)
    # plt.show()
    # crash

    if dolfin.MPI.comm_world.rank == 0:
        fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 8))
        plt.suptitle(suptitle)

    def symmetric_vrange(p):
        ignored_minp, maxp = common.minmax(p, take_abs=True, mode="raw")
        return -maxp, maxp

    if dolfin.MPI.comm_world.rank == 0:
        plt.sca(ax[0, 0])
    vmin, vmax = symmetric_vrange(fh_distorted)
    theplot = plotmagic.mpiplot(fh_distorted, show_mesh=True, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if dolfin.MPI.comm_world.rank == 0:
        plt.colorbar(theplot)
        plt.title("Input data")
        plt.axis("equal")

    if dolfin.MPI.comm_world.rank == 0:
        plt.sca(ax[0, 1])
    fdG0 = dolfin.project(fh_distorted, V0)
    vmin, vmax = symmetric_vrange(fdG0)
    theplot = plotmagic.mpiplot(fdG0, show_mesh=True, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    if dolfin.MPI.comm_world.rank == 0:
        plt.colorbar(theplot)
        plt.title("Projected to dG0")
        plt.axis("equal")

    if dolfin.MPI.comm_world.rank == 0:
        plt.sca(ax[1, 0])
    fP1 = dolfin.project(fdG0, V1)
    # fP1 = meshmagic.patch_average(fh_distorted)  # alternatively, could use this
    vmin, vmax = symmetric_vrange(fP1)
    theplot = plotmagic.mpiplot(fP1, show_mesh=True, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    if dolfin.MPI.comm_world.rank == 0:
        plt.colorbar(theplot)
        plt.title("Projected back")
        plt.axis("equal")

    if dolfin.MPI.comm_world.rank == 0:
        plt.sca(ax[1, 1])
    e = dolfin.project(fP1 - fh, V1)  # compare to clean fh
    vmin, vmax = symmetric_vrange(e)
    theplot = plotmagic.mpiplot(e, show_mesh=True, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    if dolfin.MPI.comm_world.rank == 0:
        plt.colorbar(theplot)
        plt.title("Round-tripped minus clean input")
        plt.axis("equal")

doit("With clean original")
doit("With checkerboard oscillation", error=gh)

if dolfin.MPI.comm_world.rank == 0:
    plt.show()
