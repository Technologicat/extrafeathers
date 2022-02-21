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

__all__ = ["my_cells", "all_cells", "nodes_to_array",
           "midpoint_refine", "P2_to_refined_P1",
           "mpiplot", "plot_facet_meshfunction"]

from collections import defaultdict
from enum import IntEnum
import typing

import numpy as np
import scipy.spatial.ckdtree
import matplotlib as mpl
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

import dolfin


# TODO: Split general mesh / function space utilities into `meshmagic` or some such new module.

# TODO: Does the refine belong here, or in `midpoint_refine` itself?
def my_cells(V: dolfin.FunctionSpace, *,
             matplotlibize: bool = False,
             refine_p2: bool = False) -> typing.Tuple[np.array,
                                                      typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {0: [x1, y1], ...}

    Nodal (Lagrange) elements only. Note this returns all nodes, not just the mesh vertices.
    For example, for P2 triangles, you'll get also the nodes at the midpoints of the edges.

    This only sees the cells assigned to the current MPI process; the complete function space
    is the union of these cell sets from all processes. See `all_cells`, which automatically
    combines the data from all MPI processes.

    `V`:             Scalar function space.
                       - 2D and 3D ok.
                       - `.sub(j)` of a vector/tensor function space ok.

                     Note that if `V` uses degree 1 Lagrange elements, then the output `nodes`
                     will be the vertices of the mesh. This can be useful for extracting
                     mesh data for e.g. `matplotlib`.

    `matplotlibize`: If `True`, and `V` is a 2D triangulation, ensure that the cells in the
                     output list their vertices in an anticlockwise order, as required when
                     the data is used to construct a `matplotlib.tri.Triangulation`.

    `refine_p2`:     If `True`, and `V` is a 2D P2 triangulation, split each original triangle
                     into four P1 triangles; one touching each vertex, and one in the middle,
                     spanning the edge midpoints.

                     This subtriangle arrangement looks best for visualizing P2 data,
                     when interpolating that data as P1 on the once-refined mesh. It is
                     particularly useful for export into a vertex-based format, for
                     visualization.

    Returns `(cells, nodes)`, where:
        - `cells` is a rank-2 `np.array`, with the entries global DOF numbers, one cell per row.

        - `nodes` is a `dict`, mapping from global DOF number to global node coordinates.

          In serial mode for a scalar `FunctionSpace`, the keys are `range(V.dim())`.
          In MPI mode, each process gets a different subrange due to MPI partitioning.

          Also, if `V` is a `.sub(j)` of a `VectorFunctionSpace` or `TensorFunctionSpace`,
          the DOF numbers use the *global* DOF numbering also in the sense that each
          vector/tensor component of the field maps to its own set of global DOF numbers.
    """
    if V.ufl_element().num_sub_elements() > 1:
        raise ValueError(f"Expected a scalar `FunctionSpace`, got a function space on {V.ufl_element()}")

    if not (V.mesh().topology().dim() == 2 and
            str(V.ufl_element().cell()) == "triangle"):
        matplotlibize = False

    # TODO: refine also P3 triangles in 2D?
    # TODO: refine also P2/P3 tetras in 3D?
    if not (V.mesh().topology().dim() == 2 and
            str(V.ufl_element().cell()) == "triangle" and
            V.ufl_element().degree() == 2):
        refine_p2 = False

    # "my" = local to this MPI process
    all_my_cells = []  # e.g. [[i1, i2, i3], ...] in global DOF numbering
    all_my_nodes = {}  # dict to auto-eliminate duplicates (same global DOF)
    dofmap = V.dofmap()
    element = V.element()
    l2g = dofmap.tabulate_local_to_global_dofs()
    for cell in dolfin.cells(V.mesh()):
        local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
        nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
        if not refine_p2:  # general case
            local_dofss = [local_dofs]
            nodess = [nodes]
        else:  # 2D P2 -> once-refined 2D P1
            # Split into four P1 triangles.
            #
            # - DOFs 0, 1, 2 are at the vertices
            # - DOF 3 is on the side opposite to 0
            # - DOF 4 is on the side opposite to 1
            # - DOF 5 is on the side opposite to 2
            #
            # ASCII diagram (numbering on the reference element):
            #
            # 2          2
            # |\         |\
            # 4 3   -->  4-3
            # |  \       |\|\
            # 0-5-1      0-5-1
            #  P2         4×P1
            #
            # https://fenicsproject.discourse.group/t/how-to-get-global-to-local-edge-dof-mapping-for-triangular-p2-elements/5197
            assert len(local_dofs) == 6, len(local_dofs)
            local_dofss = []
            nodess = []
            for i, j, k in ((0, 5, 4),
                            (5, 3, 4),
                            (5, 1, 3),
                            (4, 3, 2)):
                local_dofss.append([local_dofs[i], local_dofs[j], local_dofs[k]])
                nodess.append([nodes[i], nodes[j], nodes[k]])

        # Convert the constituent cells to global DOF numbering
        for local_dofs, nodes in zip(local_dofss, nodess):
            # Matplotlib wants anticlockwise ordering when building a Triangulation
            if matplotlibize and not is_anticlockwise(nodes):
                local_dofs = local_dofs[::-1]
                nodes = nodes[::-1]
                assert is_anticlockwise(nodes)

            global_dofs = l2g[local_dofs]  # [i1, i2, i3] in global numbering
            global_nodes = {ix: vtx for ix, vtx in zip(global_dofs, nodes)}  # global dof -> coordinates

            all_my_cells.append(global_dofs)
            all_my_nodes.update(global_nodes)
    return all_my_cells, all_my_nodes


def all_cells(V: dolfin.FunctionSpace, *,
              matplotlibize: bool = False,
              refine_p2: bool = False) -> typing.Tuple[np.array,
                                                       typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {0: [x1, y1], ...}

    Like `my_cells` (which see for details), but combining data from all MPI processes.
    Each process gets a full copy of all data.
    """
    cells, nodes = my_cells(V, matplotlibize=matplotlibize, refine_p2=refine_p2)
    cells = dolfin.MPI.comm_world.allgather(cells)
    nodes = dolfin.MPI.comm_world.allgather(nodes)

    # Combine the cell connectivity lists from all MPI processes.
    # The result is a single rank-2 array, with each row the global DOF numbers for a cell, e.g.:
    # [[i1, i2, i3], ...], [[j1, j2, j3], ...], ... -> [[i1, i2, i3], ..., [j1, j2, j3], ...]
    cells = np.concatenate(cells)

    # Combine the global DOF index to DOF coordinates mappings from all MPI processes.
    # After this step, each global DOF should have a corresponding vertex.
    def merge(mappings):
        merged = {}
        while mappings:
            merged.update(mappings.pop())
        return merged
    nodes = merge(nodes)
    assert len(nodes) == V.dim()  # each DOF of V has coordinates

    return cells, nodes


def nodes_to_array(nodes: typing.Dict[int, typing.List[float]]) -> typing.Tuple[np.array, np.array]:
    """Unpack the `nodes` dict return value of `my_cells` or `all_cells`.

    Returns `(dofs, nodes_array)`, where:
        - `dofs` is a rank-1 `np.array` with global DOF numbers, sorted in ascending order.

          For notes on global DOF numbers in the presence of MPI partitioning and `.sub(j)`
          of vector/tensor function spaces, see `my_cells`.

        - `nodes_array` is a rank-2 `np.array` with row `j` the coordinates for global DOF `dofs[j]`.
    """
    nodes_sorted = list(sorted(nodes.items(), key=lambda item: item[0]))  # key = global DOF number
    global_dofs = [dof for dof, ignored_node in nodes_sorted]
    node_arrays = [node for ignored_dof, node in nodes_sorted]  # list of len-2 arrays [x1, y1], [x2, y2], ...
    nodes_array = np.stack(node_arrays)  # rank-2 array [[x1, y1], [x2, y2], ...]
    return global_dofs, nodes_array


def make_mesh(cells: typing.List[typing.List[int]],
              dofs: np.array,
              vertices: np.array, *,
              distributed: bool = True) -> dolfin.Mesh:
    """Given lists of cells and vertices, build a `dolfin.Mesh`.

    `cells`:       Cell connectivity data as rank-2 `np.array`, one cell per row.
                   (This is the format returned by `all_cells`.)
    `dofs`:        Vertex numbers used by the `cells` array to refer to the vertices.
    `vertices`:    Vertex coordinates as rank-2 `np.array`, one vertex per row.
                   `vertices[k]` corresponds to vertex number `dofs[k]`.

    `distributed`:  Used when running in MPI mode:

                    If `True`, the mesh will be distributed between the MPI processes.
                    If `False`, each process constructs its own mesh independently.

    Returns a `dolfin.Mesh`.
    """
    # dolfin.MeshEditor is the right tool for the job.
    #     https://fenicsproject.org/qa/12253/integration-on-predefined-grid/
    #
    # Using it in MPI mode needs some care. Build the mesh on the root process,
    # then MPI-partition it using `dolfin.MeshPartitioning`.
    #     https://bitbucket.org/fenics-project/dolfin/issues/403/mesheditor-and-meshinit-not-initializing
    #
    if distributed:
        mesh = dolfin.Mesh()  # mesh distributed to MPI processes
    else:
        mesh = dolfin.Mesh(dolfin.MPI.comm_self)  # mesh local to each MPI process

    if not distributed or dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        editor = dolfin.MeshEditor()
        geometric_dim = np.shape(vertices)[1]
        topological_dim = geometric_dim  # TODO: get topological dimension from cell type.
        editor.open(mesh, "triangle", topological_dim, geometric_dim)  # TODO: support other cell types; look them up in the FEniCS C++ API docs.
        # Probably, one of the args is the MPI-local count and the other is the global count,
        # but this is not documented, and there is no implementation on the Python level;
        # need to look at the C++ sources of FEniCS to be sure.
        editor.init_vertices_global(len(vertices), len(vertices))
        editor.init_cells_global(len(cells), len(cells))
        for dof, vtx in zip(dofs, vertices):
            editor.add_vertex_global(dof, dof, vtx)  # local_index, global_index, coordinates
        for cell_index, triangle in enumerate(cells):
            editor.add_cell(cell_index, triangle)  # curiously, there is no add_cell_global.
        editor.close()
        mesh.init()
        mesh.order()
        # dolfin.info(mesh)  # DEBUG: distributed mode should have all mesh data on root process at this point

    if distributed:
        dolfin.MeshPartitioning.build_distributed_mesh(mesh)
        # dolfin.info(mesh)  # DEBUG: distributed mode should have distributed data to all processes now

    return mesh


def midpoint_refine(mesh: dolfin.Mesh) -> dolfin.Mesh:
    """Given a 2D triangle mesh `mesh`, return a new mesh that has additional vertices at edge midpoints.

    Like `dolfin.refine(mesh)` without further options; but we guarantee that one of the
    four subtriangles spans the edge midpoints of the original triangle.

    This subtriangle arrangement looks best for visualizing P2 data, when interpolating that
    data as P1 on the once-refined mesh.
    """
    V = dolfin.FunctionSpace(mesh, 'P', 2)            # define a scalar P2 space...
    cells, nodes_dict = all_cells(V, refine_p2=True)  # ...so that `all_cells` can do our dirty work
    dofs, nodes_array = nodes_to_array(nodes_dict)
    return make_mesh(cells, dofs, nodes_array, distributed=True)


def P2_to_refined_P1(V: typing.Union[dolfin.FunctionSpace,
                                     dolfin.VectorFunctionSpace,
                                     dolfin.TensorFunctionSpace],
                     W: typing.Union[dolfin.FunctionSpace,
                                     dolfin.VectorFunctionSpace,
                                     dolfin.TensorFunctionSpace]) -> typing.Tuple[np.array, np.array]:
    """Build a global DOF map between a P2 space `V` and a once-refined P1 space `W`.

    The purpose is to be able to map a nodal values vector from `V` to `W` and vice versa,
    for interpolation.

    Once particular use case is to export P2 data in MPI mode at full nodal resolution,
    as once-refined P1 data. In general, the MPI partitionings of `V` and `W` will
    not match, and thus `interpolate(..., W)` will not work, because each process is
    missing some of the input data needed to construct its part of the P1
    representation. This can be worked around by hacking the DOF vectors directly.
    See example below.

    `V`: P2 `FunctionSpace`, `VectorFunctionSpace`, or `TensorFunctionSpace` on some `mesh`.
    `W`: The corresponding P1 space on `refine(mesh)` or `midpoint_refine(mesh)`.

    This function is actually slightly more general than that; that is just the simple
    way to explain it. The real restriction is that `V` and `W` must have the same number
    of subspaces, if any, and that they must have the same number of global DOFs. Also,
    the corresponding nodes on `V` and `W` must be geometrically coincident.

    Returns `(VtoW, WtoV)`, where:
      - `VtoW` is a rank-1 `np.array`. Global DOF `k` of `V` matches the global DOF `VtoW[k]` of space `W`.
      - `WtoV` is a rank-1 `np.array`. Global DOF `k` of `W` matches the global DOF `WtoV[k]` of space `V`.

    Example::

        import numpy as np
        import dolfin
        from extrafeathers.plotutil import midpoint_refine, P2_to_refined_P1

        mesh = ...

        xdmffile_u = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
        xdmffile_u.parameters["flush_output"] = True
        xdmffile_u.parameters["rewrite_function_mesh"] = False

        V = dolfin.FunctionSpace(mesh, 'P', 2)
        u = dolfin.Function(V)

        export_mesh = midpoint_refine(mesh)
        W = dolfin.FunctionSpace(export_mesh, 'P', 1)
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

            # Instead, we can hack the DOF vectors directly:
            u.vector().gather(u_copy, all_V_dofs)  # actually allgather
            w.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
            # Now `w` is a refined P1 representation of the P2 field `u`.

            xdmffile_u.write(w, t)  # in parallel!
            t += dt
    """
    if W.num_sub_spaces() != V.num_sub_spaces():
        raise ValueError(f"V and W must have as many subspaces; V has {V.num_sub_spaces()}, but W has {W.num_sub_spaces()}.")
    if W.dim() != V.dim():
        raise ValueError("V and W must have as many global DOFs; V has {V.dim()} DOFs, but W has {W.dim()}.")

    # In a vector/tensor function space, each geometric node (global DOF
    # coordinates) has an independent instance for each field component,
    # so we must work one component at a time.
    if V.num_sub_spaces() > 1:
        spaces = [(V.sub(j), W.sub(j)) for j in range(V.num_sub_spaces())]
    else:
        spaces = [(V, W)]

    # Note that in MPI mode each process constructs its own full copy independently.
    VtoW = np.zeros(V.dim(), dtype=int) - 1
    WtoV = np.zeros(W.dim(), dtype=int) - 1
    seenV = set()
    seenW = set()
    for subV, subW in spaces:
        # The MPI partitionings of V and W are in general different, so to be
        # able to do the matching, we must gather the data for the whole mesh.
        ignored_cellsW, nodesW_dict = all_cells(subW)
        ignored_cellsV, nodesV_dict = all_cells(subV)
        dofsW, nodesW = nodes_to_array(nodesW_dict)
        dofsV, nodesV = nodes_to_array(nodesV_dict)

        # O(n log(n)) global geometric search, similar to what could be done
        # in serial mode using `dolfin.Mesh.bounding_box_tree`. (That function
        # itself works just fine in MPI mode; the problem are the different
        # partitionings.)
        treeV = scipy.spatial.cKDTree(data=nodesV)
        def find_nodeV_index(nodeW):
            distance, k = treeV.query(nodeW)
            if distance > 0.0:  # node dofsV[k] of V should exactly correspond to node dofW of W
                raise ValueError(f"Node {nodeW} on W and its closest neighbor {nodesV[k]} on V are not coincident.")
            return k

        for dofW, nodeW in zip(dofsW, nodesW):
            k = find_nodeV_index(nodeW)
            dofV, nodeV = dofsV[k], nodesV[k]  # noqa: F841, don't need nodeV; just for documentation
            WtoV[dofW] = dofV
            VtoW[dofV] = dofW
            seenV.add(dofV)
            seenW.add(dofW)

    # # contract: postconditions
    assert len(set(range(V.dim())) - seenV) == 0  # all dofs of V were seen
    assert len(set(range(W.dim())) - seenW) == 0  # all dofs of W were seen
    assert all(dofW != -1 for dofW in VtoW)  # each dof of V was mapped to *some* dof of W
    assert all(dofV != -1 for dofV in WtoV)  # each dof of W was mapped to *some* dof of V
    assert len(set(VtoW)) == len(VtoW)  # each dof of V was mapped to a *different* dof of W
    assert len(set(WtoV)) == len(WtoV)  # each dof of W was mapped to a *different* dof of V

    return VtoW, WtoV


def is_anticlockwise(ps: typing.List[typing.List[float]]) -> typing.Optional[bool]:
    """[[x1, y1], [x2, y2], [x3, y3]] -> whether the points are listed anticlockwise.

    In the degenerate case where the points are exactly on a line (up to machine precision),
    returns `None`.

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
def mpiplot(u: typing.Union[dolfin.Function, dolfin.Expression],
            **kwargs: typing.Any) -> typing.Optional[typing.Any]:
    """Like `dolfin.plot`, but plots the whole field in the root process (MPI rank 0).

    `u`: `dolfin.Function`; a 2D scalar FEM field
    `kwargs`: passed through to `matplotlib.pyplot.tricontourf`

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
    triangles, nodes_dict = all_cells(V_vis, matplotlibize=True, refine_p2=True)
    if my_rank == 0:
        assert len(nodes_dict) == n_global_dofs  # each global DOF has coordinates
        assert len(v_vec) == n_global_dofs  # we have a data value at each DOF
        dofs, nodes = nodes_to_array(nodes_dict)
        assert len(dofs) == n_global_dofs  # each global DOF was mapped

        # Reassemble the mesh in Matplotlib.
        # TODO: map `triangles`; it has global DOF numbers `dofs[k]`, whereas we need just `k`
        # TODO: so that the numbering corresponds to the rows of `nodes`.
        # TODO: Ignoring the mapping works as long as the data comes from a scalar `FunctionSpace`.
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=triangles)

        # Plot the solution on the mesh. The triangulation has been constructed
        # following the FEniCS global DOF numbering, so the data is just v_vec as-is.
        theplot = plt.tricontourf(tri, v_vec, levels=32, **kwargs)

        # # DEBUG
        # plt.triplot(tri, color="#404040")  # all edges
        # if V_vis.ufl_element().degree() == 2:
        #     # Original element edges for P2 element; these are the edges of the corresponding P1 triangulation.
        #     V_lin = dolfin.FunctionSpace(mesh, "P", 1)
        #     triangles2, nodes_dict2 = all_cells(V_lin, matplotlibize=True, refine_p2=True)
        #     dofs2, nodes2 = nodes_to_array(nodes_dict2)
        #     # TODO: map `triangles2`
        #     tri2 = mtri.Triangulation(nodes2[:, 0], nodes2[:, 1], triangles=triangles2)
        #     plt.triplot(tri2, color="k")

        # # Alternative visualization style.
        # theplot = plt.tripcolor(tri, v_vec, shading="gouraud", **kwargs)

        # # Another alternative visualization style.
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
