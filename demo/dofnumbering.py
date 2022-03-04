# -*- coding: utf-8; -*-
"""Investigate the global DOF numbering of P1 meshes in 2D.

Can be run serially or in MPI mode.
"""

import numpy as np
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
my_dofs = V.dofmap().dofs()  # MPI-local

# https://fenicsproject.discourse.group/t/parallel-usage-of-vertex-to-dof-map/6420/3
ownership_range = V.dofmap().ownership_range()
num_dofs_local = ownership_range[1] - ownership_range[0]
global_unowned = V.dofmap().local_to_global_unowned()
v2d = dolfin.vertex_to_dof_map(V)

print(f"rank {dolfin.MPI.comm_world.rank}: #global {V.dim()}, #local {len(v2d)}, #owned {num_dofs_local}: {my_dofs}, #unowned {len(global_unowned)}: {global_unowned}")

# --------------------------------------------------------------------------------
# plot

# Make a scalar function whose value is the global DOF number
f = dolfin.Function(V)
f.vector()[:] = my_dofs
theplot = plotmagic.mpiplot(f)


# The triangulation  uses the global DOF numbers `dofs[k]`, whereas when plotting the mesh below,
# we need just `k`, so that the numbering corresponds to the row index of `nodes`.
#
# We must do this globally, with access to the full DOF data.
ignored_cells, all_nodes_dict = meshmagic.all_cells(V)
dofs, nodes = meshmagic.nodes_to_array(all_nodes_dict)

dof_to_row_dict = {dof: k for k, dof in enumerate(dofs)}
dof_to_row_list = [v for k, v in sorted(dof_to_row_dict.items(), key=lambda item: item[0])]
dof_to_row_array = np.array(dof_to_row_list, dtype=np.uint64)

# Actually, it looks like the result is always the identity mapping, regardless of MPI group size?
# print(dolfin.MPI.comm_world.rank, dof_to_row_array)


# Now we can get the mesh parts from all processes.
# We actually only need the triangles; we can ignore the DOF/coordinate data,
# since we already have it (from the global scan).
my_cells, my_nodes_dict = meshmagic.my_cells(V)
gathered_cells = dolfin.MPI.comm_world.allgather(my_cells)

# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]

if dolfin.MPI.comm_world.rank == 0:
    for mpi_rank, cells in enumerate(gathered_cells):
        # Do the mapping, just in case.
        triangles = []
        for cell in cells:
            # print(cell, [dof_to_row_array[dof] for dof in cell])  # DEBUG
            triangles.append([dof_to_row_array[dof] for dof in cell])
        # print(triangles)  # DEBUG

        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles)
        plt.triplot(tri, color=colors[mpi_rank])

    plt.title("Global DOF number")
    plt.colorbar(theplot)
    plt.show()
