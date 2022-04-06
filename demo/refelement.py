# -*- coding: utf-8; -*-
"""Investigate the local DOF numbering on the reference triangle.

Also shows the global DOF numbers for a very small unit square mesh.

Can be run serially or in parallel:

    python -m demo.refelement
    mpirun -n 2 python -m demo.refelement
    mpirun python -m demo.refelement

You can give an element type (Px, Qx, DPx, DQx) as argument:

    python -m demo.refelement P2
    python -m demo.refelement Q2
    python -m demo.refelement DP3
    python -m demo.refelement DQ3

Node and text color color-codes cells. Labels are "reference_dof (global_dof)".

When running in parallel, line color color-codes MPI partitioning.
"""

from collections import defaultdict
from contextlib import contextmanager
import sys

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
def _colors_with_alpha(aa: str) -> list:
    return [f"{color}{aa}" for color in colors]
colors80 = _colors_with_alpha("80")

@contextmanager
def mpl_scalefont(factor: float = 2.0):
    """Context manager: temporarily change Matplotlib font size.

    Usage::

        with mpl_scalefont(1.5):
            plt.something(...)
    """
    oldsize = mpl.rcParams["font.size"]
    try:
        mpl.rcParams["font.size"] = oldsize * factor
        yield
    finally:
        mpl.rcParams["font.size"] = oldsize

# --------------------------------------------------------------------------------
# Plot

# Take element type from command-line argument if given
arg = sys.argv[1] if len(sys.argv) > 1 else "P3"
family, degree = arg[:-1], int(arg[-1])

celltype = dolfin.CellType.Type.quadrilateral if "Q" in family else dolfin.CellType.Type.triangle
mesh = dolfin.UnitSquareMesh.create(2, 2, celltype)
V = dolfin.FunctionSpace(mesh, family, degree)

dofmap = V.dofmap()
element = V.element()
l2g = dofmap.tabulate_local_to_global_dofs()  # MPI-local to global

if dolfin.MPI.comm_world.rank == 0:
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
show_partitioning = dolfin.MPI.comm_world.size > 1
plotmagic.mpiplot_mesh(V, show_partitioning=show_partitioning)

# # DEBUG
# from extrafeathers import meshmagic
# cells, nodes = meshmagic.all_cells(V)
# cells, nodes = meshmagic.quad_to_tri(cells, nodes, mpi_global=True)
# dofs, nodes_array = meshmagic.nodes_to_array(nodes)
# P1_mesh = meshmagic.make_mesh(cells, dofs, nodes_array)
# W = dolfin.FunctionSpace(P1_mesh, "P", 1)
# w = dolfin.Function(W)
# w.vector()[:] = range(W.dim())
# theplot = plotmagic.mpiplot(w)
# plt.colorbar(theplot)
# plotmagic.mpiplot_mesh(W)
# plt.axis("equal")

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
        with mpl_scalefont(1.1):
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
        # if new_y_offset > 2.9 * bbox_text.height:  # three labels per column
        #     offsets[key] = (offset[0] + bbox_text.width, 0.0)
        # else:
        #     offsets[key] = (offset[0], new_y_offset)
        offsets[key] = (offset[0], new_y_offset)  # all labels in same column

    mpi_str = f"; {dolfin.MPI.comm_world.size} MPI processes" if dolfin.MPI.comm_world.size > 1 else ""
    plt.title(f"{V.ufl_element().family()} {V.ufl_element().degree()}; {V.dim()} global DOFs on mesh{mpi_str}")
    plt.axis("off")
    plt.show()
