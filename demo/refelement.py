# -*- coding: utf-8; -*-
"""Investigate the local DOF numbering on the reference triangle.

Run serially:

    python -m demo.refelement

Can be run in parallel:

    mpirun -n 2 demo.refelement

but then each MPI process plots its own figure. (The global DOF numbers
match across the plots, as expected.)
"""

from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import dolfin

# --------------------------------------------------------------------------------
# Config

# Shift the label slightly so that it doesn't overlap the node indicator.
text_x_offset = 0.02  # in data coordinates

# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]

# --------------------------------------------------------------------------------
# Plot

mesh = dolfin.UnitSquareMesh(1, 1)
V = dolfin.FunctionSpace(mesh, "P", 3)

dofmap = V.dofmap()
element = V.element()
l2g = dofmap.tabulate_local_to_global_dofs()

fig, ax = plt.subplots(1, 1, constrained_layout=True)
dolfin.plot(mesh, color="#c0c0c0")

y_offsets = defaultdict(lambda: 0.0)
for cell in dolfin.cells(V.mesh()):
    nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
    local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
    global_dofs = l2g[local_dofs]  # [i1, i2, i3] in global numbering

    for reference_dof, (node, global_dof) in enumerate(zip(nodes, global_dofs)):
        y_offset = y_offsets[global_dof]
        label = f"[{reference_dof}] {global_dof}" if not y_offset else f"[{reference_dof}]"
        plt.plot(node[0], node[1], color="k", marker="o")
        text = plt.text(node[0] + text_x_offset, node[1] - y_offset, label,
                        color=colors[cell.index() % len(colors)],
                        horizontalalignment="left", verticalalignment="bottom")

        # Increase offset by one line, for next element that shares this global DOF.
        # On getting the text display size in data coordinates in matplotlib, see:
        #     https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
        renderer = fig.canvas.get_renderer()
        bbox_text = text.get_window_extent(renderer=renderer)
        bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))  # to data coordinates
        y_offsets[global_dof] += bbox_text.height

mpi_rank_str = f"MPI rank {dolfin.MPI.comm_world.rank}: " if dolfin.MPI.comm_world.size > 1 else ""
plt.title(f"{mpi_rank_str}DOF numbering (P{V.ufl_element().degree()} elements), notation: [ref] global")
plt.axis("off")
plt.show()
