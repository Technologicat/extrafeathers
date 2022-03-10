# -*- coding: utf-8; -*-
"""Patch-averaging for P1 functions.

Works both serially and in MPI mode.

    python -m demo.patch_average
    mpirun python -m demo.patch_average
"""

import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin
import extrafeathers

N = 4
mesh = dolfin.UnitSquareMesh(N, N)
V = dolfin.FunctionSpace(mesh, 'P', 1)
f = dolfin.Expression('sin(N/4 * 2 * pi * x[0])', degree=2, N=N)
g = dolfin.project(f, V)

# TODO: need a utility for this. `demo.dofnumbering` needs something similar, too.
def mpiplot_mesh():
    cells, nodes_dict = extrafeathers.meshmagic.all_cells(V)
    dofs, nodes = extrafeathers.meshmagic.nodes_to_array(nodes_dict)
    if dolfin.MPI.comm_world.rank == 0:
        cells = np.array(cells, dtype=np.int64)
        # This relies on the fact that for p>1, in FEniCS the vertices are the first DOFs in each triangle.
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=cells[:, :3])
        plt.triplot(tri, color="#a0a0a040")

if dolfin.MPI.comm_world.rank == 0:
    plt.subplot(2, 1, 1)
    plt.title("Patch-averaging demo")
theplot = extrafeathers.mpiplot(g)
mpiplot_mesh()
if dolfin.MPI.comm_world.rank == 0:
    plt.ylabel("Original")
    plt.colorbar(theplot)
    plt.subplot(2, 1, 2)
theplot = extrafeathers.mpiplot(extrafeathers.patch_average(g))
mpiplot_mesh()
if dolfin.MPI.comm_world.rank == 0:
    plt.ylabel("Patch-averaged")
    plt.colorbar(theplot)
    plt.show()
