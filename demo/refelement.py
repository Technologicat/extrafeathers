# -*- coding: utf-8; -*-
"""Investigate the local DOF numbering on the reference triangle.

Run serially:

    python -m demo.refelement
"""

from collections import defaultdict
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import dolfin

from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Config

# Shift the label slightly so that it doesn't overlap the node indicator.
text_x_offset = 0.03  # in data coordinates

# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]

# Mix translucent versions. `colors` must be in "#rrggbb" format for this to work.
def _colors_with_alpha(aa):
    return [f"{color}{aa}" for color in colors]
colors80 = _colors_with_alpha("80")

@contextmanager
def mpl_largefont():
    oldsize = mpl.rcParams["font.size"]
    try:
        mpl.rcParams["font.size"] = oldsize * 2
        yield
    finally:
        mpl.rcParams["font.size"] = oldsize

# --------------------------------------------------------------------------------
# Plot

# TODO: parameterize element type (command-line argument, argparse)
mesh = dolfin.UnitSquareMesh(1, 1)
V = dolfin.FunctionSpace(mesh, "DP", 3)

dofmap = V.dofmap()
element = V.element()
l2g = dofmap.tabulate_local_to_global_dofs()

fig, ax = plt.subplots(1, 1, constrained_layout=True)
plotmagic.mpiplot_mesh(V, show_partitioning=True, linewidth=3.0)

# TODO: MPI mode

y_offsets = defaultdict(lambda: 0.0)
for cell in reversed(list(dolfin.cells(V.mesh()))):
    nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
    local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
    global_dofs = l2g[local_dofs]  # [i1, i2, i3] in global numbering

    for reference_dof, (node, global_dof) in enumerate(zip(nodes, global_dofs)):
        y_offset = y_offsets[tuple(nodes[reference_dof])]
        label = f"{reference_dof} #{global_dof}"  # if not y_offset else f"[{reference_dof}]"
        plt.plot(node[0], node[1], colors80[cell.index() % len(colors)],
                 marker="o", markersize=10.0)
        with mpl_largefont():
            text = plt.text(node[0] + text_x_offset, node[1] - y_offset, label,
                            color=colors[cell.index() % len(colors)],
                            horizontalalignment="left",
                            verticalalignment="center_baseline")

        # Increase offset by one line, for next element that shares this global DOF.
        # On getting the text display size in data coordinates in matplotlib, see:
        #     https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
        renderer = fig.canvas.get_renderer()
        bbox_text = text.get_window_extent(renderer=renderer)
        bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))  # to data coordinates
        y_offsets[tuple(nodes[reference_dof])] += bbox_text.height

mpi_rank_str = f"MPI rank {dolfin.MPI.comm_world.rank}: " if dolfin.MPI.comm_world.size > 1 else ""
with mpl_largefont():
    plt.title(f"{mpi_rank_str}{V.ufl_element().family()} {V.ufl_element().degree()}")
plt.axis("off")
plt.show()
