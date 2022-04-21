# -*- coding: utf-8; -*-
"""Investigate the global DOF numbering of P1, P2 and P3 meshes in 2D.

Can be run serially or in MPI mode.

From the top-level directory of the project:

    python -m demo.dofnumbering
    mpirun -n 2 python -m demo.dofnumbering
    mpirun -n 4 python -m demo.dofnumbering

Element type can also be given:

    python -m demo.dofnumbering Q2
    mpirun -n 2 python -m demo.dofnumbering Q2
    mpirun -n 4 python -m demo.dofnumbering Q2

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

# Take element type from command-line argument if given
arg = sys.argv[1] if len(sys.argv) > 1 else "P2"
family, degree = arg[:-1], int(arg[-1])

# S elements are not supported by FFC (at least in FEniCS 2019), but let's be correct here.
# Note that `extrafeathers.plotmagic` does not (yet) support S elements, either.
celltype = dolfin.CellType.Type.quadrilateral if ("Q" in family or "S" in family) else dolfin.CellType.Type.triangle
mesh = dolfin.UnitSquareMesh.create(N, N, celltype)
V = dolfin.FunctionSpace(mesh, family, degree)

# --------------------------------------------------------------------------------
# Print some stats

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

if dolfin.MPI.comm_world.rank == 0:
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))

# Plot a scalar function whose value is the global DOF number.
f = dolfin.Function(V)
f.vector()[:] = my_owned
theplot = plotmagic.mpiplot(f)
if dolfin.MPI.comm_world.rank == 0:
    plt.colorbar(theplot)

# Plot the mesh parts.
plotmagic.mpiplot_mesh(V, show_partitioning=True)

if dolfin.MPI.comm_world.rank == 0:
    mpi_str = f"; {dolfin.MPI.comm_world.size} MPI processes" if dolfin.MPI.comm_world.size > 1 else ""
    plt.title(f"{V.ufl_element().family()} {V.ufl_element().degree()}; {V.dim()} global DOFs on mesh{mpi_str}")
    plt.legend(loc="upper right")  # show the labels for the mesh parts
    plt.show()
