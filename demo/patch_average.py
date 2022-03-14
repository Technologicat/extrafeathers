# -*- coding: utf-8; -*-
"""Patch-averaging for P1 functions.

Works both serially and in MPI mode.

    python -m demo.patch_average
    mpirun python -m demo.patch_average
"""

import matplotlib.pyplot as plt

import dolfin
import extrafeathers

# --------------------------------------------------------------------------------
# Demo

N = 4
mesh = dolfin.UnitSquareMesh(N, N)

# patch-average a scalar field
V = dolfin.FunctionSpace(mesh, 'P', 1)
f = dolfin.Expression('1 - (4*pow(x[0] - 0.5, 2) + 4*pow(x[1] - 0.5, 2))', degree=2, N=N)
g = dolfin.project(f, V)
g_pavg = extrafeathers.patch_average(g)

# patch-average a vector field
W = dolfin.VectorFunctionSpace(mesh, 'P', 1)
Wproj = dolfin.VectorFunctionSpace(mesh, 'DG', 0)
WtoWproj, Wprojtocell = extrafeathers.map_dG0(W, Wproj)
cell_volume = extrafeathers.cellvolume(W.mesh())
f2 = dolfin.Expression(('sin(N/4 * 2 * pi * x[0])',
                        'sin(N/4 * 2 * pi * x[1])'), degree=2, N=N)
g2 = dolfin.project(f2, W)
g2_pavg = extrafeathers.patch_average(g2, Wproj, WtoWproj, Wprojtocell, cell_volume)

# --------------------------------------------------------------------------------
# Visualize

# Scalar
if dolfin.MPI.comm_world.rank == 0:
    plt.figure(1)
    plt.subplot(2, 1, 1)
theplot = extrafeathers.mpiplot(g, show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.axis("equal")
    plt.ylabel("Original")
    plt.colorbar(theplot)
    plt.subplot(2, 1, 2)
theplot = extrafeathers.mpiplot(g_pavg, show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.axis("equal")
    plt.ylabel("Patch-averaged")
    plt.colorbar(theplot)
    plt.suptitle("Patch-averaging demo (scalar)")

# Vector
if dolfin.MPI.comm_world.rank == 0:
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.title(r"$g_1$")
theplot = extrafeathers.mpiplot(g2.sub(0), show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.ylabel("Original")
    plt.colorbar(theplot)
    plt.subplot(2, 2, 2)
    plt.title(r"$g_2$")
theplot = extrafeathers.mpiplot(g2.sub(1), show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)
    plt.subplot(2, 2, 3)
theplot = extrafeathers.mpiplot(g2_pavg.sub(0), show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.ylabel("Patch-averaged")
    plt.colorbar(theplot)
    plt.subplot(2, 2, 4)
theplot = extrafeathers.mpiplot(g2_pavg.sub(1), show_mesh=True)
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)
    plt.suptitle("Patch-averaging demo (vector)")

if dolfin.MPI.comm_world.rank == 0:
    plt.show()
