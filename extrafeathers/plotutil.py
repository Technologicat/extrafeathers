# -*- coding: utf-8; -*-
"""Plotting utilities for FEniCS, using `matplotlib` as the backend.

We provide some "extra batteries" (i.e. features oddly missing from FEniCS itself)
for some common plotting tasks in 2D solvers:

- `mpiplot` allows a solver on a triangle mesh to plot the *whole* solution
  in the root process while running in MPI mode. This is often useful for
  debugging and visualizing simulation progress, especially for light MPI jobs
  that run locally on a laptop.

- `plot_facet_meshfunction` does exactly as it says on the tin. This is often
  useful for debugging and visualization when generating and importing meshes;
  it allows to check whether the boundaries have been tagged as expected.
"""

__all__ = ["my_triangles", "all_triangles", "mpiplot", "plot_facet_meshfunction"]

from collections import defaultdict
from enum import IntEnum
import typing

import numpy as np
import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin


def my_triangles(V: dolfin.FunctionSpace) -> typing.Tuple[typing.List[typing.List[int]],
                                                          typing.Dict[int, typing.List[float]]]:
    """P1 FunctionSpace -> triangle connectivity [[i1, i2, i3], ...], vertices {0: [x1, y1], ...}

    Only sees the triangles assigned to the current MPI process; but FEniCS partitions the cells
    without overlaps, so the complete mesh is the union of these triangle sets from all processes.

    See `all_triangles`, which combines the data from different MPI processes so you get
    exactly what it says on the tin.
    """
    if V.mesh().topology().dim() != 2 or V.ufl_element().degree() > 1 or str(V.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"This function only supports meshes of topological dimension 2, with degree 1 triangle elements, got a mesh of topological dimension {V.mesh().topology().dim()} with degree {V.ufl_element().degree()} {V.ufl_element().cell()} elements.")
    if V.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"This function only supports meshes of geomertric dimension 2, got a mesh of geometric dimension {V.mesh().geometric_dimension()}.")

    # "my" = local to this MPI process
    all_my_global_indices = []
    all_my_vertices = {}  # dict to auto-eliminate duplicates (same global DOF)
    l2g = V.dofmap().tabulate_local_to_global_dofs()
    element = V.element()  # TODO: what if P1 VectorFunctionSpace? (for now, scalars only)
    dofmap = V.dofmap()
    for cell in dolfin.cells(V.mesh()):
        local_dof_indices = dofmap.cell_dofs(cell.index())  # local to this MPI process
        vertices = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]]

        # TODO: support P2 triangles.
        # The cell and edge midpoints tell us which DOFs are on the edge or in the middle;
        # any remaining ones are at the vertices of the triangle. We can then convert each
        # P2 element into a patch of linear triangles for visualization.

        # Matplotlib wants anticlockwise ordering when building a Triangulation
        if not is_anticlockwise(vertices):
            local_dof_indices = local_dof_indices[::-1]
            vertices = vertices[::-1]
        assert is_anticlockwise(vertices)

        global_dof_indices = l2g[local_dof_indices]  # [i1, i2, i3] in global numbering
        global_dof_to_vertex = {ix: vtx for ix, vtx in zip(global_dof_indices, vertices)}

        all_my_global_indices.append(global_dof_indices)
        all_my_vertices.update(global_dof_to_vertex)
    return all_my_global_indices, all_my_vertices


def all_triangles(V: dolfin.FunctionSpace) -> typing.Tuple[typing.List[typing.List[int]],
                                                           typing.Dict[int, typing.List[float]]]:
    """P1 FunctionSpace -> triangle connectivity [[i1, i2, i3], ...], vertices {0: [x1, y1], ...}

    Combines data from all MPI processes. Each process gets a copy of the complete triangulation.
    """
    if V.mesh().topology().dim() != 2 or V.ufl_element().degree() > 1 or str(V.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"This function only supports meshes of topological dimension 2, with degree 1 triangle elements, got a mesh of dimension {V.mesh().topology().dim()} with degree {V.ufl_element().degree()} {V.ufl_element().cell()} elements.")
    if V.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"This function only supports meshes of geomertric dimension 2, got a mesh of geometric dimension {V.mesh().geometric_dimension()}.")

    ixs, vtxs = my_triangles(V)
    ixs = dolfin.MPI.comm_world.allgather(ixs)
    vtxs = dolfin.MPI.comm_world.allgather(vtxs)

    # Combine the triangle connectivity lists from all MPI processes.
    ixs = np.concatenate(ixs)  # [[i1, i2, i3], ...], [[j1, j2, j3], ...], ... -> [[i1, i2, i3], ...]

    # Combine the global DOF index to vertex coordinates mappings from all MPI processes.
    # After this step, each global DOF should have a corresponding vertex.
    merged = vtxs.pop()
    for vtx in vtxs:
        merged.update(vtx)
    vtxs = merged

    # List the vertices in global DOF numbering order.
    # When plotting, this allows us to use the DOF vector for the field data as-is.
    vtx_sorted_by_global_dof = sorted(vtxs.items(), key=lambda item: item[0])  # key = global DOF number
    vtxs = [vtx for ignored_ix, vtx in vtx_sorted_by_global_dof]
    vtxs = np.stack(vtxs)  # list of len-2 arrays [x1, y1], [x2, y2], ... -> array [[x1, y1], [x2, y2], ...]

    return ixs, vtxs


def is_anticlockwise(ps: typing.List[typing.List[float]]) -> typing.Optional[bool]:
    """[[x1, y1], [x2, y2], [x3, y3]] -> whether the points are listed anticlockwise.

    Based on the shoelace formula:
        https://en.wikipedia.org/wiki/Shoelace_formula
    """
    x1, y1 = ps[0]
    x2, y2 = ps[1]
    x3, y3 = ps[2]
    # https://math.stackexchange.com/questions/1324179/how-to-tell-if-3-connected-points-are-connected-clockwise-or-counter-clockwise
    s = x1 * y2 - x1 * y3 + y1 * x3 - y1 * x2 + x2 * y3 - y2 * x3
    if s > 0:
        return True  # anticlockwise
    elif s < 0:
        return False  # clockwise
    return None  # degenerate case; the points are on a line


# TODO: not sure what exactly `matplotlib.pyplot.tricontourf` returns or what the type spec for it should be.
# The point of the Optional return value is to make it explicit it's something-or-None.
def mpiplot(u: typing.Union[dolfin.Function, dolfin.Expression]) -> typing.Optional[typing.Any]:
    """Like `dolfin.plot`, but plots the whole field in the root process (MPI rank 0).

    When running serially, delegates to `dolfin.plot`.

    u: `dolfin.Function`; a 2D scalar FEM field

    In the root process (MPI rank 0), returns the plot object.
    See the return value of `matplotlib.pyplot.tricontourf`.

    In other processes, returns `None`.
    """
    V = u.ufl_function_space()
    mesh = V.mesh()
    my_rank = dolfin.MPI.comm_world.rank

    if mesh.topology().dim() != 2:
        raise NotImplementedError(f"mpiplot currently only supports meshes of topological dimension 2, got {mesh.topology().dim()}")
    if dolfin.MPI.comm_world.size == 1:  # running serially
        return dolfin.plot(u)

    # https://fenicsproject.discourse.group/t/gather-function-in-parallel-error/1114

    # # global DOF distribution between the MPI processes
    # d = V.dofmap().dofs()  # local, each process gets its own values
    # print(my_rank, min(d), max(d))

    # Project to P1 elements for easy reconstruction for visualization.
    if V.ufl_element().degree() > 1 or str(V.ufl_element().cell()) != "triangle":
        # if my_rank == 0:
        #     print(f"Interpolating solution from {str(V.ufl_element())} to P1 triangles for MPI-enabled visualization.")
        V_vis = dolfin.FunctionSpace(mesh, "P", 1)
        u_vis = dolfin.interpolate(u, V_vis)
    else:
        V_vis = V
        u_vis = u

    # make a complete copy of the DOF vector onto the root process
    v_vec = u_vis.vector().gather_on_zero()
    n_global_dofs = len(v_vec)

    # # make a complete copy of the DOF vector u_vec to all MPI processes
    # u_vec = u.vector()
    # v_vec = dolfin.Vector(dolfin.MPI.comm_self)  # local vector (local to each MPI process)
    # u_vec.gather(v_vec, np.array(range(V.dim()), "intc"))  # in_vec.gather(out_vec, indices); really "allgather"
    # dm = np.array(V.dofmap().dofs())
    # print(f"Process {my_rank}: local #DOFs {len(u_vec)} (min {min(dm)}, max {max(dm)}) out of global {len(v_vec)}")

    # # make a copy of the local part (in each MPI process) of u_vec only
    # u_vec = u.vector()
    # v_vec = dolfin.Vector(dolfin.MPI.comm_self, u_vec.local_size())
    # u_vec.gather(v_vec, V.dofmap().dofs())  # in_vec.gather(out_vec, indices)

    # Assemble the complete mesh from the partitioned pieces. This treats arbitrary domain shapes correctly.
    # We get the list of triangles from each MPI process and then combine the lists in the root process.
    ixs, vtxs = all_triangles(V_vis)
    if my_rank == 0:
        assert len(vtxs) == n_global_dofs

        # Reassemble the mesh in Matplotlib.
        tri = mtri.Triangulation(vtxs[:, 0], vtxs[:, 1], triangles=ixs)

        # Plot the solution on the mesh. The triangulation has been constructed
        # following the FEniCS global DOF numbering, so the data is just v_vec as-is.
        theplot = plt.tricontourf(tri, v_vec, levels=32)

        # Alternative visualization.
        # # https://matplotlib.org/stable/gallery/mplot3d/trisurf3d.html
        # ax = plt.figure().add_subplot(projection="3d")
        # theplot = ax.plot_trisurf(xs, ys, v_vec)

        return theplot
    return None


# Use `matplotlib`'s default color sequence.
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]
def plot_facet_meshfunction(f: dolfin.MeshFunction,
                            names: typing.Optional[IntEnum] = None,
                            invalid_values: typing.Optional[typing.List[int]] = None) -> None:
    """Plot a `size_t` meshfunction defined on facets of a 2D mesh.

    Useful for checking whether boundaries have been tagged as expected.

    `dolfin.plot` should be preferred, but as of FEniCS 2019, it does not support
    plotting a mesh function defined on facets.

    Colors follow `matplotlib`'s default color cycle, with the tag value 0 mapped
    to the zeroth color.

    No MPI support - for use in serial mode only.

    `f`: Mesh function of type `size_t` on facets of a mesh. Any facet for which `f` has
         a nonzero value will be plotted, and colored according to the value of `f`.
         The colors follow the default color cycle of Matplotlib.

    `names`: If provided, names for the integer values are looked up in this `IntEnum`,
             and the lines are labeled (so that `matplotlib.pyplot.legend` can then be
             used to see which is which).

             Any value of the `MeshFunction` that does not have a corresponding entry
             in `names` is ignored (for example, internal facets inside the domain).

             Thus, only facets whose tags are *whitelisted* by presence in `names` will be plotted.
             They will be labeled (for `legend`) as e.g. "INLET (ID#1)" where "INLET" is a name
             from `names`, and `1` is the corresponding tag value.

    `invalid_values`: Alternative for `names`.

                     If provided, these tag values will be ignored. Useful values:
                         [0] for a manually generated `MeshFunction`, and
                         [2**64 - 1] for Gmsh import via `meshio`.

                     Thus, all facets whose tags are *not blacklisted* by presence in `invalid_values`
                     will be plotted. They will be labeled (for `legend`) as "<boundary> (ID#X)",
                     where "X" is the tag value.

                     If `names` is provided, it takes precedence.

    No return value.
    """
    mesh = f.mesh()
    if mesh.topology().dim() != 2:
        raise NotImplementedError(f"This function only supports meshes of topological dimension 2, got {mesh.topology().dim()}")

    # Simplifying assumption: in geometric dimension 2, we can just discard the third coordinate of the vertices.
    if mesh.geometric_dimension() != 2:
        raise NotImplementedError(f"This function only supports meshes of geometric dimension 2, got {mesh.geometric_dimension()}")

    if f.dim() != 1:
        raise NotImplementedError(f"This function only supports mesh functions on facets (dimension 1); got a function of dimension {f.dim()}")
    if dolfin.MPI.comm_world.size > 1:
        # TODO: add MPI support.
        # Like the MPI plotter above, we should gather all data to the root process.
        # Not that important to implement, though, because mesh generation and import
        # (visualizing which is the primary use case for this function) is often done in serial mode.
        raise NotImplementedError("Facet meshfunction plotting currently only supported in serial mode.")

    if names:
        tag_to_name = {item.value: item.name for item in names}

    def empty_list() -> typing.List:
        return []
    plot_data = defaultdict(empty_list)
    for facet in dolfin.facets(mesh):
        tag = f[facet]
        ignore_tag = (names and tag not in tag_to_name) or (invalid_values is not None and tag in invalid_values)
        if not ignore_tag:
            vtxs = [vtx.point().array()[:2] for vtx in dolfin.vertices(facet)]  # [[x1, y1], [x2, y2]]
            plot_data[tag].append(vtxs)
            # Insert a NaN entry to force matplotlib to draw each facet separately,
            # instead of connecting them. (They are not in any useful order, and in general,
            # facets with the same tag need not form a connected line.)
            # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
            plot_data[tag].append(np.array([[np.nan, np.nan]]))
    plot_data = {tag: np.concatenate(vtxss) for tag, vtxss in sorted(plot_data.items(), key=lambda item: item[0])}

    for tag, vtxs in plot_data.items():
        label = f"{tag_to_name[tag]} (ID#{tag})" if names else f"<boundary> (ID#{tag})"
        plt.plot(vtxs[:, 0], vtxs[:, 1], color=colors[tag % len(colors)], label=label)
