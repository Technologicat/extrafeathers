# -*- coding: utf-8; -*-
"""Patch-averaging for P1 functions.

Works both serially and in MPI mode.

    python -m demo.patch_average
    mpirun python -m demo.patch_average
"""

import matplotlib.pyplot as plt

import dolfin
import extrafeathers

N = 4
mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.FunctionSpace(mesh, 'P', 1)
f = dolfin.Expression('sin(N/4 * 2 * pi * x[0])', degree=2, N=N)
g = dolfin.project(f, V)

if dolfin.MPI.comm_world.rank == 0:
    plt.subplot(2, 1, 1)
theplot = extrafeathers.mpiplot(g)
if dolfin.MPI.comm_world.rank == 0:
    plt.title("Original function")
    plt.colorbar(theplot)
    plt.subplot(2, 1, 2)
theplot = extrafeathers.mpiplot(extrafeathers.patch_average(g))
if dolfin.MPI.comm_world.rank == 0:
    plt.title("Patch average")
    plt.colorbar(theplot)
    plt.show()
