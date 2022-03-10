# -*- coding: utf-8; -*-
"""Investigate the global DOF numbering of P1, P2 and P3 meshes in 2D.

Can be run serially or in MPI mode.

From the top-level directory of the project:

    python -m demo.dofnumbering
    mpirun -n 2 python -m demo.dofnumbering
    mpirun -n 4 python -m demo.dofnumbering

The results present in a visual form how FEnICS aims to maximize
data locality when numbering DOFs in MPI mode.
"""

import sys

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Set up the mesh

N = 8
mesh = dolfin.UnitSquareMesh(N, N)

# --------------------------------------------------------------------------------
# Print some stats

V = dolfin.FunctionSpace(mesh, "P", 2)  # try P1, P2 or P3 elements here

# MPI-local, containing global DOF numbers
# See also:
#   https://fenicsproject.discourse.group/t/parallel-usage-of-vertex-to-dof-map/6420/3
my_owned = V.dofmap().dofs()
my_unowned = V.dofmap().local_to_global_unowned()
my_total = len(my_owned) + len(my_unowned)

totals = dolfin.MPI.comm_world.allgather(my_total)

# global stats
if dolfin.MPI.comm_world.rank == 0:
    nglobal = V.dim()  # number of distinct global DOFs
    nprocs = dolfin.MPI.comm_world.size
    ntotal = sum(totals)
    print(f"{nprocs} processes, DOFS #global {nglobal}, Σ(#local) {ntotal}, overhead {100 * (ntotal / nglobal - 1):0.3g}%")

sys.stdout.flush()
dolfin.MPI.comm_world.barrier()  # always print the global stats first

# local stats
# print(f"MPI rank {dolfin.MPI.comm_world.rank}: #local {my_total}, #owned {len(my_owned)}: {my_owned}, #unowned {len(my_unowned)}: {my_unowned}")
print(f"MPI rank {dolfin.MPI.comm_world.rank}: #local {my_total}, #owned {len(my_owned)}, #unowned {len(my_unowned)}")

# --------------------------------------------------------------------------------
# Plot

# Plot a scalar function whose value is the global DOF number.
f = dolfin.Function(V)
f.vector()[:] = my_owned
theplot = plotmagic.mpiplot(f)
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)

# Plot the mesh parts.
plotmagic.mpiplot_mesh(V, show_partitioning=True)

if dolfin.MPI.comm_world.rank == 0:
    plt.title(f"Global DOF number (P{V.ufl_element().degree()} elements)")
    plt.legend(loc="upper right")  # show the labels for the mesh parts
    plt.show()
