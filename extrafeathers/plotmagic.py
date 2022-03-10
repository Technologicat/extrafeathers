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

__all__ = ["pause",
           "mpiplot", "mpiplot_mesh",
           "plot_facet_meshfunction"]

from collections import defaultdict
from enum import IntEnum
import typing

import numpy as np
import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin

from .meshmagic import all_cells, nodes_to_array


def pause(interval: float) -> None:
    """Redraw the current Matplotlib figure **without stealing focus**.

    **IMPORTANT**:

    Works after `plt.show()` has been called at least once.

    **Background**:

    Matplotlib (3.3.3) has a habit of popping the figure window to top when it
    is updated using show() or pause(), which effectively prevents using the
    machine for anything else while a simulation is in progress.

    On some systems, it helps to force Matplotlib to use the "Qt5Agg" backend:
        https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus

    but on some others (Linux Mint 20.1) also that backend steals focus.

    One option is to use a `FuncAnimation`, but it has a different API from
    just plotting regularly, and it is not applicable in all cases.

    So, we provide this a custom non-focus-stealing pause function hack,
    based on the StackOverflow answer by user @ImportanceOfBeingErnest:
        https://stackoverflow.com/a/45734500

    """
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)


# TODO: not sure what exactly `matplotlib.pyplot.tricontourf` returns or what the type spec for it should be.
# The point of the Optional return value is to make it explicit it's something-or-None.
def mpiplot(u: typing.Union[dolfin.Function, dolfin.Expression],
            *,
            show_mesh: bool = False,
            **kwargs: typing.Any) -> typing.Optional[typing.Any]:
    """Like `dolfin.plot`, but plots the whole field in the root process (MPI rank 0).

    `u`: `dolfin.Function`; a 2D scalar FEM field
         (`.sub(j)` of a vector or tensor field is fine)
    `show_mesh`: if `True`, show the element edges (and P1 vis edges too,
                 if `u` is a `P2` or `P3` function).
    `kwargs`: passed through to `matplotlib.pyplot.tricontourf`

    If `u` lives on P2 or P3 elements, each element will be internally split into
    P1 triangles for visualization.

    In the root process (MPI rank 0), returns the plot object.
    See the return value of `matplotlib.pyplot.tricontourf`.

    In other processes, returns `None`.
    """
    V = u.function_space()
    mesh = V.mesh()
    my_rank = dolfin.MPI.comm_world.rank

    if mesh.topology().dim() != 2:
        raise NotImplementedError(f"mpiplot currently only supports meshes of topological dimension 2, got {mesh.topology().dim()}")

    # https://fenicsproject.discourse.group/t/gather-function-in-parallel-error/1114

    # # global DOF distribution between the MPI processes
    # d = V.dofmap().dofs()  # local, each process gets its own values
    # print(my_rank, min(d), max(d))

    # Project to P1 elements for easy reconstruction for visualization.
    if V.ufl_element().degree() not in (1, 2, 3) or str(V.ufl_element().cell()) != "triangle":
        # if my_rank == 0:
        #     print(f"Interpolating solution from {str(V.ufl_element())} to P1 triangles for MPI-enabled visualization.")
        V_vis = dolfin.FunctionSpace(mesh, "P", 1)
        u_vis = dolfin.interpolate(u, V_vis)
    else:
        V_vis = V
        u_vis = u

    # Make a complete copy of the DOF vector onto the root process.
    v_vec = u_vis.vector().gather_on_zero()
    n_global_dofs = V_vis.dim()

    # # make a complete copy of the DOF vector u_vec to all MPI processes
    # u_vec = u.vector()
    # v_vec = dolfin.Vector(dolfin.MPI.comm_self)  # local vector (local to each MPI process)
    # u_vec.gather(v_vec, np.array(range(V.dim()), "intc"))  # in_vec.gather(out_vec, indices); really "allgather"
    # dm = np.array(V.dofmap().dofs())
    # print(f"Process {my_rank}: local #DOFs {len(dm)} (min {min(dm)}, max {max(dm)}) out of global {V.dim()}")

    # # make a copy of the local part (in each MPI process) of u_vec only
    # u_vec = u.vector()
    # v_vec = dolfin.Vector(dolfin.MPI.comm_self, u_vec.local_size())
    # u_vec.gather(v_vec, V.dofmap().dofs())  # in_vec.gather(out_vec, indices)

    # The global DOF vector always refers to the complete function space.
    # If `V` is a subspace (vector/tensor field component), the DOF vector
    # will include also those DOFs that are not part of `V`. Extract the
    # V DOFs.
    dofmaps = dolfin.MPI.comm_world.gather(V_vis.dofmap().dofs(), root=0)
    if my_rank == 0:
        # CAUTION: `all_cells` sorts its `dofs` by global DOF number; match the ordering.
        subspace_dofs = np.sort(np.concatenate(dofmaps))
        v_vec = v_vec[subspace_dofs]

    # Assemble the complete mesh from the partitioned pieces. This treats arbitrary
    # domain shapes correctly. We get the list of triangles from each MPI process
    # and then combine the lists in the root process.
    triangles, nodes_dict = all_cells(V_vis, matplotlibize=True, refine=True)
    if my_rank == 0:
        assert len(nodes_dict) == n_global_dofs  # each global DOF has coordinates
        assert len(v_vec) == n_global_dofs  # we have a data value at each DOF
        dofs, nodes = nodes_to_array(nodes_dict)
        assert len(dofs) == n_global_dofs  # each global DOF was mapped

        # Reassemble the mesh in Matplotlib.

        # Map `triangles`; it has global DOF numbers `dofs[k]`, whereas we need just `k`
        # so that the numbering corresponds to the rows of `nodes`.
        #
        # Ignoring the mapping works as long as the data comes from a scalar
        # `FunctionSpace`; then it's the identity mapping. But with subspace data
        # we need to do this.
        #
        # Note that to perform this mapping we need a full copy of the global DOF
        # data, because the triangulation uses also the unowned nodes. Thus the
        # MPI-local part is not enough.
        dof_to_row = {dof: k for k, dof in enumerate(dofs)}  # `dofs` is already sorted by global DOF number
        new_triangles = [[dof_to_row[dof] for dof in triangle]
                         for triangle in triangles]

        # Now we can construct the triangulation.
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=new_triangles)

        # Plot the function, and optionally also the mesh.
        theplot = plt.tricontourf(tri, v_vec, levels=32, **kwargs)
        if show_mesh:
            mpiplot_mesh(V_vis, _triangulation=tri)

        # # Alternative visualization style.
        # theplot = plt.tripcolor(tri, v_vec, shading="gouraud", **kwargs)

        # # Another alternative visualization style.
        # # https://matplotlib.org/stable/gallery/mplot3d/trisurf3d.html
        # ax = plt.figure().add_subplot(projection="3d")
        # theplot = ax.plot_trisurf(xs, ys, v_vec)

        return theplot
    return None


def mpiplot_mesh(V: dolfin.FunctionSpace, *,
                 main_color: str = "#80808040",
                 aux_color: str = "#c0c0c040",
                 show_aux: bool = True,
                 _triangulation=None) -> typing.Optional[typing.Any]:
    """Plot the mesh of a `FunctionSpace`.

    2D triangle meshes only.

    Like `dolfin.plot(mesh)`, but plots the whole mesh in the root process.
    Also, is able to show the P1 vis edges (generated by `extrafeathers`)
    if `V` is a P2 or P3 `FunctionSpace`.

    `main_color`: "#RRGGBBAA", element edges.
    `aux_color`: "#RRGGBBAA", internal edges for P1 vis refinements.
                 Used only if `V` is a P2 or P3 function space.
    `show_aux`: Whether to plot the P1 vis refinement edges if present.

    `_triangulation` is an internal parameter used by `mpiplot`, so that
    the `show_mesh` mode can re-use the triangulation `mpiplot` must build
    anyway to visualize the actual function. If interested, see the source
    code of `mpiplot_mesh` and `mpiplot`.

    In the root process (MPI rank 0), returns the plot object for the
    element edges. See the return value of `matplotlib.pyplot.triplot`.

    In other processes, returns `None`.
    """
    # See `mpiplot` for detailed explanation. We essentially re-do the
    # relevant parts here so that this function also works standalone.
    if not _triangulation:
        triangles, nodes_dict = all_cells(V, matplotlibize=True, refine=True)
        if dolfin.MPI.comm_world.rank == 0:
            dofs, nodes = nodes_to_array(nodes_dict)
            dof_to_row = {dof: k for k, dof in enumerate(dofs)}
            new_triangles = [[dof_to_row[dof] for dof in triangle]
                             for triangle in triangles]
            _triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1],
                                                triangles=new_triangles)

    main_plot = None
    all_edges_color = aux_color if V.ufl_element().degree() > 1 else main_color
    if dolfin.MPI.comm_world.rank == 0:
        if show_aux or V.ufl_element().degree() <= 1:
            main_plot = plt.triplot(_triangulation, color=all_edges_color)
    if V.ufl_element().degree() > 1:
        # Original element edges for P2/P3 element, i.e. the edges of the
        # corresponding P1 triangulation.
        V_lin = dolfin.FunctionSpace(V.mesh(), "P", 1)
        triangles2, nodes_dict2 = all_cells(V_lin, matplotlibize=True)
        dofs2, nodes2 = nodes_to_array(nodes_dict2)
        if dolfin.MPI.comm_world.rank == 0:
            # `V_lin` is always scalar, so `triangles2` needs no DOF->row mapping.
            tri2 = mtri.Triangulation(nodes2[:, 0], nodes2[:, 1],
                                      triangles=triangles2)
            main_plot = plt.triplot(tri2, color=main_color)
    return main_plot


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
