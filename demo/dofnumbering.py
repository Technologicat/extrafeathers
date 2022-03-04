# -*- coding: utf-8; -*-
"""Investigate the global DOF numbering of P1 meshes in 2D.

Can be run serially or in MPI mode.
"""

import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin

from extrafeathers import meshmagic
from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Set up the mesh

N = 4
mesh = dolfin.UnitSquareMesh(N, N)

# --------------------------------------------------------------------------------
# print some stats

V = dolfin.FunctionSpace(mesh, "P", 1)

# MPI-local, containing global DOF numbers
# See also:
#   https://fenicsproject.discourse.group/t/parallel-usage-of-vertex-to-dof-map/6420/3
my_owned = V.dofmap().dofs()
my_unowned = V.dofmap().local_to_global_unowned()
my_total = len(my_owned) + len(my_unowned)

print(f"rank {dolfin.MPI.comm_world.rank}: #global {V.dim()}, #local {my_total}, #owned {len(my_owned)}: {my_owned}, #unowned {len(my_unowned)}: {my_unowned}")

# --------------------------------------------------------------------------------
# plot

# Make a scalar function whose value is the global DOF number
f = dolfin.Function(V)
f.vector()[:] = my_owned
theplot = plotmagic.mpiplot(f)

# The triangulation for the mesh parts uses the global DOF numbers `dofs[k]`,
# and contains also the unowned nodes. Thus we must get a copy of the full
# global DOF and coordinate data to be able to plot the mesh parts.
ignored_cells, all_nodes_dict = meshmagic.all_cells(V)
dofs, nodes = meshmagic.nodes_to_array(all_nodes_dict)

# Get the mesh parts from all processes.
# We actually only need the triangles; we can ignore the DOF/coordinate data,
# since we already have a full copy (from the global scan).
my_cells, my_nodes_dict = meshmagic.my_cells(V)
gathered_cells = dolfin.MPI.comm_world.allgather(my_cells)

# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]

if dolfin.MPI.comm_world.rank == 0:
    for mpi_rank, cells in enumerate(gathered_cells):
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=cells)
        plt.triplot(tri, color=colors[mpi_rank])

    plt.title("Global DOF number")
    plt.colorbar(theplot)
    plt.show()
