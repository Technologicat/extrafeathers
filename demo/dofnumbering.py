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

import numpy as np
import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin

from extrafeathers import meshmagic
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

# The triangulation for the mesh parts uses the global DOF numbers `dofs[k]`,
# but refers also to the unowned nodes. Thus we must get a copy of the full
# global DOF and coordinate data to be able to plot the mesh parts.
ignored_cells, all_nodes_dict = meshmagic.all_cells(V)
dofs, nodes = meshmagic.nodes_to_array(all_nodes_dict)

# Get the mesh parts from all processes. We actually only need the triangles;
# we can ignore the partial DOF/coordinate data, since we already have a full
# copy (from the global scan).
my_cells, my_nodes_dict = meshmagic.my_cells(V)
gathered_cells = dolfin.MPI.comm_world.allgather(my_cells)

# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]

# Plot the mesh parts, color-coding the MPI processes.
# TODO: refactor this as an option into `plotmagic.mpiplot_mesh`
if dolfin.MPI.comm_world.rank == 0:
    for mpi_rank, cells in enumerate(gathered_cells):
        cells = np.array(cells, dtype=np.int64)
        # This relies on the fact that for p>1, in FEniCS the vertices are the first DOFs in each triangle.
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=cells[:, :3])
        plt.triplot(tri, color=colors[mpi_rank])

    plt.title(f"Global DOF number (P{V.ufl_element().degree()} elements)")
    plt.colorbar(theplot)

    # Each legend entry from `triplot` is doubled for some reason,
    # so plot a dummy point with each of the line colors and label them.
    for mpi_rank in range(dolfin.MPI.comm_world.size):
        plt.plot([0], [0], color=colors[mpi_rank], label=f"MPI rank {mpi_rank}")
    plt.legend(loc="upper right")

    plt.show()
