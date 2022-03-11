# -*- coding: utf-8; -*-
"""Investigate the local DOF numbering on the reference triangle.

Also shows the global DOF numbers for a very small unit square mesh.

Can be run serially or in parallel:

    python -m demo.refelement
    mpirun -n 2 python -m demo.refelement
    mpirun python -m demo.refelement

Line color is for MPI partitioning; node and text color is for cells.
"""

from collections import defaultdict
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from unpythonic import flatten1

import dolfin

from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Config

# Shift the label slightly so that it doesn't overlap the node indicator.
text_padding_left = 0.02  # in data coordinates

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
mesh = dolfin.UnitSquareMesh(2, 2)
V = dolfin.FunctionSpace(mesh, "DP", 3)
if dolfin.MPI.comm_world.rank == 0:
    print(f"Total global #DOFS: {V.dim()}")

dofmap = V.dofmap()
element = V.element()
l2g = dofmap.tabulate_local_to_global_dofs()  # MPI-local to global

if dolfin.MPI.comm_world.rank == 0:
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
plotmagic.mpiplot_mesh(V, linewidth=3.0, show_partitioning=True)

data = []
for cell in dolfin.cells(V.mesh()):
    nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
    local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
    global_dofs = l2g[local_dofs]  # [i1, i2, i3] in global numbering

    for reference_dof, (node, global_dof) in enumerate(zip(nodes, global_dofs)):
        data.append((cell.global_index(), node, reference_dof, global_dof))

data = dolfin.MPI.comm_world.gather(data, root=0)
if data:
    data = flatten1(data)

if dolfin.MPI.comm_world.rank == 0:
    offsets = defaultdict(lambda: (0.0, 0.0))
    for cell_index, node, reference_dof, global_dof in data:
        key = tuple(node)
        offset = offsets[key]
        label = f"{reference_dof} ({global_dof})"  # if not offset else f"[{reference_dof}]"
        plt.plot(node[0], node[1], colors80[cell_index % len(colors)],
                 marker="o", markersize=10.0)
        text = plt.text(node[0] + offset[0] + text_padding_left,
                        node[1] - offset[1], label,  # +y up so next line is in the -y direction
                        color=colors[cell_index % len(colors)],
                        horizontalalignment="left",
                        verticalalignment="bottom")

        # Increase offset by one line, for next element that shares this global DOF.
        # On getting the text display size in data coordinates in matplotlib, see:
        #     https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
        renderer = fig.canvas.get_renderer()
        bbox_text = text.get_window_extent(renderer=renderer)
        bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))  # to data coordinates
        new_y_offset = offset[1] + bbox_text.height
        if new_y_offset > 2.9 * bbox_text.height:
            offsets[key] = (offset[0] + bbox_text.width, 0.0)
        else:
            offsets[key] = (offset[0], offset[1] + bbox_text.height)

    mpi_str = f" (with {dolfin.MPI.comm_world.size} MPI processes)" if dolfin.MPI.comm_world.size > 1 else ""
    plt.title(f"{V.ufl_element().family()} {V.ufl_element().degree()}{mpi_str}")
    plt.axis("off")
    plt.show()
