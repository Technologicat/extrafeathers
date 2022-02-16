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

__all__ = ["my_triangles", "all_triangles", "sort_vtxs", "P2_to_refined_P1",
           "mpiplot", "plot_facet_meshfunction"]

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
    """P1 or P2 FunctionSpace -> triangle connectivity [[i1, i2, i3], ...], coordinates {0: [x1, y1], ...}

    2D triangle meshes only.

    Only sees the triangles assigned to the current MPI process; but FEniCS partitions the cells
    without overlaps, so the complete mesh is the union of these triangle sets from all processes.

    See `all_triangles`, which combines the data from different MPI processes so you get
    exactly what it says on the tin.

    P2 triangles are internally split into four P1 triangles; one touching each vertex,
    and one in the middle, consisting of the edge midpoints.

    If the data comes from `.sub(j)` of a `VectorFunctionSpace`, the keys of the `dict` use the
    *global* DOF numbering, with each vector component mapping to a different set of DOF numbers.
    """
    if V.mesh().topology().dim() != 2 or V.ufl_element().degree() not in (1, 2) or str(V.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"This function only supports meshes of topological dimension 2, with degree 1 or 2 triangle elements, got a mesh of topological dimension {V.mesh().topology().dim()} with {V.ufl_element()}.")
    if V.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"This function only supports meshes of geometric dimension 2, got a mesh of geometric dimension {V.mesh().geometric_dimension()}.")
    if V.ufl_element().num_sub_elements() > 1:
        raise ValueError(f"Expected a scalar `FunctionSpace`, got a function space on {V.ufl_element()}")

    # "my" = local to this MPI process
    all_my_triangles = []  # [[i1, i2, i3], ...] in global DOF numbering
    all_my_dof_coordinates = {}  # dict to auto-eliminate duplicates (same global DOF)
    l2g = V.dofmap().tabulate_local_to_global_dofs()
    element = V.element()  # TODO: what if P1 VectorFunctionSpace? (for now, scalars only)
    dofmap = V.dofmap()
    for cell in dolfin.cells(V.mesh()):
        # Split the element into constituent triangles on which we can linearly interpolate for visualization
        local_dof_indices = dofmap.cell_dofs(cell.index())  # local to this MPI process
        dof_coordinates = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]]
        if V.ufl_element().degree() == 1:  # P1
            if len(local_dof_indices) != 3:
                raise ValueError(f"Expected a scalar `FunctionSpace`, but this P1 triangle has {len(local_dof_indices)} local DOFs. If it is a `VectorFunctionSpace`, consider taking its `.sub(j)`.")
            # Just one triangle - the element itself.
            local_dof_indicess = [local_dof_indices]
            dof_coordinatess = [dof_coordinates]
        else:  # V.ufl_element().degree() == 2:  # P2
            if len(local_dof_indices) != 6:
                raise ValueError(f"Expected a scalar `FunctionSpace`, but this P2 triangle has {len(local_dof_indices)} local DOFs. If it is a `VectorFunctionSpace`, consider taking its `.sub(j)`.")
            # Split into four P1 triangles.
            # https://fenicsproject.discourse.group/t/how-to-get-global-to-local-edge-dof-mapping-for-triangular-p2-elements/5197
            #
            # - DOFs 0, 1, 2 are at the vertices
            # - DOF 3 is on the side opposite to 0
            # - DOF 4 is on the side opposite to 1
            # - DOF 5 is on the side opposite to 2
            #
            # ASCII diagram (numbering on the reference element):
            #
            # 2
            # |\
            # 4 3
            # |  \
            # 0-5-1
            local_dof_indicess = []
            dof_coordinatess = []
            for i, j, k in ((0, 5, 4),
                            (5, 3, 4),
                            (5, 1, 3),
                            (4, 3, 2)):
                local_dof_indicess.append([local_dof_indices[i], local_dof_indices[j], local_dof_indices[k]])
                dof_coordinatess.append([dof_coordinates[i], dof_coordinates[j], dof_coordinates[k]])

        # Convert the constituent triangles to global DOF numbering
        for local_dof_indices, dof_coordinates in zip(local_dof_indicess, dof_coordinatess):
            # Matplotlib wants anticlockwise ordering when building a Triangulation
            if not is_anticlockwise(dof_coordinates):
                local_dof_indices = local_dof_indices[::-1]
                dof_coordinates = dof_coordinates[::-1]
            assert is_anticlockwise(dof_coordinates)

            global_dof_indices = l2g[local_dof_indices]  # [i1, i2, i3] in global numbering
            global_dof_to_coordinates = {ix: vtx for ix, vtx in zip(global_dof_indices, dof_coordinates)}

            all_my_triangles.append(global_dof_indices)
            all_my_dof_coordinates.update(global_dof_to_coordinates)
    return all_my_triangles, all_my_dof_coordinates


def all_triangles(V: dolfin.FunctionSpace) -> typing.Tuple[np.array,
                                                           typing.Dict[int, typing.List[float]]]:
    """P1 or P2 FunctionSpace -> triangle connectivity [[i1, i2, i3], ...], coordinates {0: [x1, y1], ...}

    2D triangle meshes only.

    Combines data from all MPI processes. Each process gets a copy of the complete triangulation.

    P2 triangles are internally split into four P1 triangles; one touching each vertex,
    and one in the middle, consisting of the edge midpoints.

    If the data comes from `.sub(j)` of a `VectorFunctionSpace`, the keys of the `dict` use the
    *global* DOF numbering, with each vector component mapping to a different set of DOF numbers.
    """
    if V.mesh().topology().dim() != 2 or V.ufl_element().degree() not in (1, 2) or str(V.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"This function only supports meshes of topological dimension 2, with degree 1 or 2 triangle elements, got a mesh of dimension {V.mesh().topology().dim()} with {V.ufl_element()}.")
    if V.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"This function only supports meshes of geometric dimension 2, got a mesh of geometric dimension {V.mesh().geometric_dimension()}.")
    if V.ufl_element().num_sub_elements() > 1:
        raise ValueError(f"Expected a scalar `FunctionSpace`, got a function space on {V.ufl_element()}")

    triangles, vtxs = my_triangles(V)
    triangles = dolfin.MPI.comm_world.allgather(triangles)
    vtxs = dolfin.MPI.comm_world.allgather(vtxs)

    # Combine the triangle connectivity lists from all MPI processes.
    # The result is a single rank-2 array, with each row the global DOF numbers for a triangle:
    # [[i1, i2, i3], ...], [[j1, j2, j3], ...], ... -> [[i1, i2, i3], ..., [j1, j2, j3], ...]
    triangles = np.concatenate(triangles)

    # Combine the global DOF index to DOF coordinates mappings from all MPI processes.
    # After this step, each global DOF should have a corresponding vertex.
    merged = vtxs.pop()
    for vtx in vtxs:
        merged.update(vtx)
    vtxs = merged

    return triangles, vtxs


def sort_vtxs(vtxs: typing.Dict[int, typing.List[float]]) -> typing.Tuple[np.array, np.array]:
    """List the global DOF coordinates in ascending order of global DOF number.

    `vtxs`: as returned by `my_triangles` or `all_triangles`

    Returns `(dofs, vtxs)`, where:
        - `dofs` is a rank-1 `np.array` with global DOF numbers, sorted in ascending order.

          On a scalar function space, this is essentially just `np.arange(V.dim())`, but
          if the data comes from `.sub(j)` of a `VectorFunctionSpace`, this is the *global*
          DOF numbering, with each vector component mapping to a different set of DOF numbers.

        - `vtxs` is a rank-2 `np.array` with row `j` the coordinates for global DOF `dofs[j]`

    When plotting, this allows using the DOF vector for scalar field data as-is.
    """
    vtx_sorted_by_global_dof = list(sorted(vtxs.items(), key=lambda item: item[0]))  # key = global DOF number
    dofs = [ix for ix, ignored_vtx in vtx_sorted_by_global_dof]
    vtxs = [vtx for ignored_ix, vtx in vtx_sorted_by_global_dof]
    vtxs = np.stack(vtxs)  # list of len-2 arrays [x1, y1], [x2, y2], ... -> array [[x1, y1], [x2, y2], ...]
    return dofs, vtxs


def P2_to_refined_P1(V, W):
    """Map global DOFs for exporting P2 data at full nodal resolution, as once-refined P1 data.

    This is useful in MPI mode, where the partitionings of `V` and `W` will not match,
    and thus `interpolate(..., W)` will not work, as each process is missing some of
    the input data needed to construct its part of the P1 function. See example below.

    2D triangle meshes only.

    Slow for large meshes; based on an O(n²) geometric search for the matching nodes.

    `V`: P2 `FunctionSpace`, `VectorFunctionSpace`, or `TensorFunctionSpace` on some `mesh`.
    `W`: The corresponding P1 space on `refine(mesh)`.

    Returns `(VtoW, WtoV)`, where:
      - `VtoW` is a rank-1 `np.array`. Global DOF `k` of `V` is the global DOF `VtoW[k]` of space `W`.
      - `WtoV` is a rank-1 `np.array`. Global DOF `k` of `W` is the global DOF `WtoV[k]` of space `V`.

    You can use this DOF mapping data to map a nodal values vector from `V` to `W` and vice versa.

    Example::

        import numpy as np
        import dolfin
        from extrafeathers.plotutil import P2_to_refined_P1

        mesh = ...

        xdmffile_u = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
        xdmffile_u.parameters["flush_output"] = True
        xdmffile_u.parameters["rewrite_function_mesh"] = False

        V = dolfin.FunctionSpace(mesh, 'P', 2)
        u = dolfin.Function(V)

        mesh_refined = dolfin.refine(mesh)
        W = dolfin.FunctionSpace(mesh_refined, 'P', 1)
        w = dolfin.Function(W)

        VtoW, WtoV = P2_to_refined_P1(V, W)

        all_V_dofs = np.array(range(V.dim()), "intc")
        u_copy = dolfin.Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
        my_W_dofs = W.dofmap().dofs()  # MPI-local
        my_V_dofs = WtoV[my_W_dofs]  # MPI-local

        ...  # set up PDE problem

        T = ...
        nt = ...
        dt = T / nt

        t = 0
        for n in range(nt):
            ...
            dolfin.solve(..., u.vector(), ...)

            # This is what we want to do, and it works in serial mode;
            # but not in MPI mode, because V and W partition differently:
            #   w.assign(interpolate(u, W))

            # So we do this instead:
            u.vector().gather(u_copy, all_V_dofs)  # actually allgather
            w.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
            # Now `w` is a refined P1 representation of the P2 field `u`.

            xdmffile_u.write(w, t)
            t += dt
    """
    if V.mesh().topology().dim() != 2 or V.ufl_element().degree() != 2 or str(V.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"V must be on a mesh of topological dimension 2, with degree 2 triangle elements; it is on a mesh of dimension {V.mesh().topology().dim()} with {V.ufl_element()}.")
    if V.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"V must be on a mesh of geometric dimension 2; it is on a mesh of geometric dimension {V.mesh().geometric_dimension()}.")
    if W.mesh().topology().dim() != 2 or W.ufl_element().degree() != 1 or str(W.ufl_element().cell()) != "triangle":
        raise NotImplementedError(f"W must be on a mesh of topological dimension 2, with degree 1 triangle elements; it is on a mesh of dimension {V.mesh().topology().dim()} with {V.ufl_element()}.")
    if W.mesh().geometric_dimension() != 2:
        raise NotImplementedError(f"W must be on a mesh of geometric dimension 2; it is on a mesh of geometric dimension {V.mesh().geometric_dimension()}.")

    # If vector/tensor function spaces, both should have the same number of vector/tensor components.
    if W.num_sub_spaces() != V.num_sub_spaces():
        raise ValueError(f"V and W must have the same number of subspaces; V has {V.num_sub_spaces()}, whereas W has {W.num_sub_spaces()}.")

    if W.dim() != V.dim():
        raise ValueError("V and W must have the same number of global DOFs; V has {V.dim()}, whereas W has {W.dim()}.")

    VtoW = np.zeros(V.dim(), dtype=int) - 1
    WtoV = np.zeros(W.dim(), dtype=int) - 1
    seenV = set()
    seenW = set()

    # Handle vector/tensor function spaces, too.
    if V.num_sub_spaces() > 1:
        spaces = [(V.sub(j), W.sub(j)) for j in range(V.num_sub_spaces())]
    else:
        spaces = [(V, W)]

    # TODO: There must be a better way to construct the DOF mapping than this woeful O(n²) geometry-based search.
    for subV, subW in spaces:
        # The `extrafeathers` plot utilities provide the full geometry data globally across MPI processses.
        # We need the global DOF coordinates and their global DOF numbers; we can ignore the triangle connectivity.
        trianglesV_ignored, vtxsV = all_triangles(subV)
        dofsV, vtxsV = sort_vtxs(vtxsV)
        trianglesW_ignored, vtxsW = all_triangles(subW)
        dofsW, vtxsW = sort_vtxs(vtxsW)
        for dofV, vtxV in zip(dofsV, vtxsV):
            # find matching node
            distances = np.sum((vtxsW - vtxV)**2, axis=1)
            k = np.argmin(distances)

            # postcondition: exact geometric match due to how the spaces were chosen
            assert distances[k] == 0.0  # node dofsW[k] of W corresponds to node dofV of V

            dofW, vtxW = dofsW[k], vtxsW[k]  # noqa: F841, don't need vtxW; just for documentation
            VtoW[dofV] = dofW
            WtoV[dofW] = dofV

            seenV.add(dofV)
            seenW.add(dofW)
    # contract: postconditions
    assert len(set(range(V.dim())) - seenV) == 0  # all dofs of V were seen
    assert len(set(range(W.dim())) - seenW) == 0  # all dofs of W were seen
    assert all(dofW != -1 for dofW in VtoW)  # each dof of V was mapped to some dof of W
    assert all(dofV != -1 for dofV in WtoV)  # each dof of W was mapped to some dof of V

    return VtoW, WtoV


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

    u: `dolfin.Function`; a 2D scalar FEM field

    If `u` lives on P2 elements, each element will be internally split into
    four P1 triangles for visualization.

    In the root process (MPI rank 0), returns the plot object.
    See the return value of `matplotlib.pyplot.tricontourf`.

    In other processes, returns `None`.
    """
    V = u.ufl_function_space()
    mesh = V.mesh()
    my_rank = dolfin.MPI.comm_world.rank

    if mesh.topology().dim() != 2:
        raise NotImplementedError(f"mpiplot currently only supports meshes of topological dimension 2, got {mesh.topology().dim()}")

    # # We do the hifi P2->P1 mapping, so let's not delegate even in serial mode.
    # if dolfin.MPI.comm_world.size == 1:  # running serially
    #     return dolfin.plot(u)

    # https://fenicsproject.discourse.group/t/gather-function-in-parallel-error/1114

    # # global DOF distribution between the MPI processes
    # d = V.dofmap().dofs()  # local, each process gets its own values
    # print(my_rank, min(d), max(d))

    # Project to P1 elements for easy reconstruction for visualization.
    if V.ufl_element().degree() not in (1, 2) or str(V.ufl_element().cell()) != "triangle":
        # if my_rank == 0:
        #     print(f"Interpolating solution from {str(V.ufl_element())} to P1 triangles for MPI-enabled visualization.")
        V_vis = dolfin.FunctionSpace(mesh, "P", 1)
        u_vis = dolfin.interpolate(u, V_vis)
    else:
        V_vis = V
        u_vis = u

    # make a complete copy of the DOF vector onto the root process
    v_vec = u_vis.vector().gather_on_zero()
    n_global_dofs = V.dim()

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

    # Assemble the complete mesh from the partitioned pieces. This treats arbitrary domain shapes correctly.
    # We get the list of triangles from each MPI process and then combine the lists in the root process.
    triangles, vtxs = all_triangles(V_vis)
    if my_rank == 0:
        assert len(vtxs) == n_global_dofs  # each global DOF has coordinates
        assert len(v_vec) == n_global_dofs  # we have a data value at each DOF
        dofs, vtxs = sort_vtxs(vtxs)
        assert len(dofs) == n_global_dofs  # each global DOF was mapped

        # Reassemble the mesh in Matplotlib.
        # TODO: map `triangles`; it has global DOF numbers `dofs[k]`, whereas we need just `k`
        # TODO: so that the numbering corresponds to the rows of `vtxs`.
        # TODO: Ignoring the mapping works as long as the data comes from a scalar `FunctionSpace`.
        tri = mtri.Triangulation(vtxs[:, 0], vtxs[:, 1], triangles=triangles)

        # Plot the solution on the mesh. The triangulation has been constructed
        # following the FEniCS global DOF numbering, so the data is just v_vec as-is.
        theplot = plt.tricontourf(tri, v_vec, levels=32)

        # # DEBUG
        # plt.triplot(tri, color="#404040")  # all edges
        # if V_vis.ufl_element().degree() == 2:
        #     # Original element edges for P2 element; these are the edges of the corresponding P1 triangulation.
        #     V_lin = dolfin.FunctionSpace(mesh, "P", 1)
        #     triangles2, vtxs2 = all_triangles(V_lin)
        #     dofs2, vtxs2 = sort_vtxs(vtxs2)
        #     # TODO: map `triangles2`
        #     tri2 = mtri.Triangulation(vtxs2[:, 0], vtxs2[:, 1], triangles=triangles2)
        #     plt.triplot(tri2, color="k")

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
