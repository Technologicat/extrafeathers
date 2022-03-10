# -*- coding: utf-8; -*-
"""Mesh-handling utilities.

Currently, some of these are useful for MPI parallel solvers, and some for
full nodal resolution export of data on P2 or P3 triangle meshes.
"""

__all__ = ["my_cells", "all_cells", "nodes_to_array",
           "make_mesh",
           "midpoint_refine", "map_refined_P1",
           "my_patches", "all_patches", "map_dG0", "patch_average"]

from collections import defaultdict
import typing

import numpy as np
import scipy.spatial.ckdtree

import dolfin

from .common import is_anticlockwise


# TODO: Does the refine belong here, or in `midpoint_refine` itself?
def my_cells(V: dolfin.FunctionSpace, *,
             matplotlibize: bool = False,
             refine: bool = False) -> typing.Tuple[np.array,
                                                   typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {0: [x1, y1], ...}

    Nodal (Lagrange) elements only. Note this returns all nodes, not just the
    mesh vertices. For example, for P2 triangles, you'll get also the nodes at
    the midpoints of the edges.

    This only sees the cells assigned to the current MPI process; the complete
    function space is the union of these cell sets from all processes. See
    `all_cells`, which automatically combines the data from all MPI processes.

    `V`:             Scalar function space.
                       - 2D and 3D ok.
                       - `.sub(j)` of a vector/tensor function space ok.

                     Note that if `V` uses degree 1 Lagrange elements, then the
                     output `nodes` will be the vertices of the mesh. This can
                     be useful for extracting mesh data for e.g. `matplotlib`.

    `matplotlibize`: If `True`, and `V` is a 2D triangulation, ensure that the
                     cells in the output list their vertices in an
                     anticlockwise order, as required when the data is used to
                     construct a `matplotlib.tri.Triangulation`.

                     Only makes sense when the output is P1.

    `refine`:        If `True`, and `V` is a 2D P2 or P3 triangulation, split each
                     original triangle into P1 triangles (four for P2, nine for P3).

                     The subtriangles are arranged in an aesthetically pleasing
                     pattern, intended for visualizing P2 or P3 data, when
                     interpolating that data as P1 on the once-refined mesh. It is
                     particularly useful for export into a vertex-based format, for
                     visualization.

    Returns `(cells, nodes)`, where:
        - `cells` is a rank-2 `np.array`, with the entries global DOF numbers,
          one cell per row.

        - `nodes` is a `dict`, mapping from global DOF number to global node coordinates.

          In serial mode for a scalar `FunctionSpace`, the keys are `range(V.dim())`.
          In MPI mode, each process gets a different subrange due to MPI partitioning.

          Also, if `V` is a `.sub(j)` of a `VectorFunctionSpace` or
          `TensorFunctionSpace`, the DOF numbers use the *global* DOF numbering
          also in the sense that each vector/tensor component of the field maps
          to its own set of global DOF numbers.

    """
    if V.ufl_element().num_sub_elements() > 1:
        raise ValueError(f"Expected a scalar `FunctionSpace`, got a function space on {V.ufl_element()}")

    input_degree = V.ufl_element().degree()
    if not (V.mesh().topology().dim() == 2 and
            str(V.ufl_element().cell()) == "triangle"):
        matplotlibize = False
    if input_degree > 1 and not refine:
        matplotlibize = False  # only P1 output can be matplotlibized

    # TODO: refine also P2/P3 tetras in 3D?
    if not (V.mesh().topology().dim() == 2 and
            str(V.ufl_element().cell()) == "triangle" and
            input_degree > 1):
        refine = False  # only P2 and P3 triangles can be refined by this function

    # "my" = local to this MPI process
    all_my_cells = []  # e.g. [[i1, i2, i3], ...] in global DOF numbering
    all_my_nodes = {}  # dict to auto-eliminate duplicates (same global DOF)
    dofmap = V.dofmap()
    element = V.element()
    l2g = dofmap.tabulate_local_to_global_dofs()
    for cell in dolfin.cells(V.mesh()):
        local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
        nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
        if not refine:  # general case
            local_dofss = [local_dofs]
            nodess = [nodes]
        elif input_degree == 2:  # 2D P2 -> once-refined 2D P1
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
            # See `demo.refelement`, and:
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
        elif input_degree == 3:  # 2D P3 -> once-refined 2D P1
            # Split into nine P1 triangles.
            #
            # - DOFs 0, 1, 2 are at the vertices
            # - DOFs 3, 4 are on the side opposite to 0
            # - DOFs 5, 6 are on the side opposite to 1
            # - DOFs 7, 8 are on the side opposite to 2
            # - DOF 8 is at the center of the triangle
            #
            # ASCII diagram (numbering on the reference element):
            #
            #       2
            #      /|
            #     6 4
            #    /  |
            #   5 9 3
            #  /    |
            # 0-7-8-1
            #
            # See `demo.refelement`.
            assert len(local_dofs) == 10, len(local_dofs)
            local_dofss = []
            nodess = []
            for i, j, k in ((0, 5, 7), (5, 7, 9), (7, 8, 9), (3, 8, 9), (1, 3, 8),
                            (5, 6, 9), (4, 6, 9), (3, 4, 9),
                            (2, 4, 6)):
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
              refine: bool = False) -> typing.Tuple[np.array,
                                                    typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {0: [x1, y1], ...}

    Like `my_cells` (which see for details), but combining data from all MPI processes.
    Each process gets a full copy of all data.
    """
    cells, nodes = my_cells(V, matplotlibize=matplotlibize, refine=refine)
    cells = dolfin.MPI.comm_world.allgather(cells)
    nodes = dolfin.MPI.comm_world.allgather(nodes)

    # Combine the cell connectivity lists from all MPI processes.
    # The result is a single rank-2 array, with each row the global DOF numbers for a cell, e.g.:
    # in:  [[i1, i2, i3], ...], [[j1, j2, j3], ...], ...
    # out: [[i1, i2, i3], ..., [j1, j2, j3], ...]
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
        - `dofs` is a rank-1 `np.array` with global DOF numbers, sorted in ascending
          order.

          For notes on global DOF numbers in the presence of MPI partitioning and
          `.sub(j)` of vector/tensor function spaces, see `my_cells`.

        - `nodes_array` is a rank-2 `np.array` with row `j` the coordinates for
          global DOF `dofs[j]`.
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

                   You can get the `dofs` and `vertices` arrays by
                   `dofs, vertices = nodes_to_array(nodes_dict)`,
                   where `nodes_dict` is the dictionary part of the
                   return value of `all_cells` (for a P1 mesh).

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
        # Probably, one of the args is the MPI-local count and the other is the global
        # count, but this is not documented, and there is no implementation on the
        # Python level; need to look at the C++ sources of FEniCS to be sure.
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


def midpoint_refine(mesh: dolfin.Mesh, p: int = 2) -> dolfin.Mesh:
    """Given a 2D triangle mesh `mesh`, return a new mesh for exporting P2 or P3 data as P1.

    Like `dolfin.refine(mesh)` without further options; but we guarantee an aesthetically optimal fill,
    designed for visualizing P2/P3 data, when interpolating that data as P1 on the once-refined mesh.

    `p`: The original degree of the elements your data lives on.
       `p=2` will generate a P1 mesh for P2 data, with an additional vertex placed at each edge midpoint.
       `p=3` will generate a P1 mesh for P3 data, with two additional vertices placed per edge, and one
             additional vertex placed in the interior of each triangle.
    """
    if p not in (2, 3):
        raise ValueError(f"Expected p = 2 or 3, got {p}.")
    V = dolfin.FunctionSpace(mesh, 'P', p)         # define a scalar P2/P3 space...
    cells, nodes_dict = all_cells(V, refine=True)  # ...so that `all_cells` can do our dirty work
    dofs, nodes_array = nodes_to_array(nodes_dict)
    return make_mesh(cells, dofs, nodes_array, distributed=True)


def map_refined_P1(V: typing.Union[dolfin.FunctionSpace,
                                   dolfin.VectorFunctionSpace,
                                   dolfin.TensorFunctionSpace],
                   W: typing.Union[dolfin.FunctionSpace,
                                   dolfin.VectorFunctionSpace,
                                   dolfin.TensorFunctionSpace]) -> typing.Tuple[np.array, np.array]:
    """Build a global DOF map between a P2 or P3 space `V` and a once-refined P1 space `W`.

    The purpose is to be able to map a nodal values vector from `V` to `W` and
    vice versa, for interpolation.

    Once particular use case is to export P2/P3 data in MPI mode at full nodal resolution,
    as once-refined P1 data. In general, the MPI partitionings of `V` and `W` will
    not match, and thus `interpolate(..., W)` will not work, because each process is
    missing some of the input data needed to construct its part of the P1
    representation. This can be worked around by hacking the DOF vectors directly.
    See example below.

    `V`: P2 or P3 `FunctionSpace`, `VectorFunctionSpace`, or `TensorFunctionSpace`
         on some `mesh`.
    `W`: The corresponding P1 space on `refine(mesh)` or `midpoint_refine(mesh)`.

    This function is actually slightly more general than that; that is just the simple
    way to explain it. The real restriction is that `V` and `W` must have the same number
    of subspaces (vector/tensor components), if any, and that they must have the same
    number of global DOFs. Also, the corresponding nodes on `V` and `W` must be
    exactly geometrically coincident (up to machine precision).

    Returns `(VtoW, WtoV)`, where:
      - `VtoW` is a rank-1 `np.array`. Global DOF `k` of `V` matches the global DOF
        `VtoW[k]` of space `W`.
      - `WtoV` is a rank-1 `np.array`. Global DOF `k` of `W` matches the global DOF
        `WtoV[k]` of space `V`.

    Example::

        import numpy as np
        import dolfin
        from extrafeathers import midpoint_refine, map_refined_P1

        mesh = ...

        xdmffile_u = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
        xdmffile_u.parameters["flush_output"] = True
        xdmffile_u.parameters["rewrite_function_mesh"] = False

        V = dolfin.FunctionSpace(mesh, 'P', 2)
        u = dolfin.Function(V)

        export_mesh = midpoint_refine(mesh)
        W = dolfin.FunctionSpace(export_mesh, 'P', 1)
        w = dolfin.Function(W)

        VtoW, WtoV = map_refined_P1(V, W)

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


def my_patches(V: dolfin.FunctionSpace) -> typing.Dict[int, np.array]:
    """Return a dict of rank-1 np.arrays, mapping DOF number to connected cell numbers.

    Only DOFs belonging to this MPI process are listed. See `all_patches`.
    """
    dofmap = V.dofmap()
    l2g = dofmap.tabulate_local_to_global_dofs()
    dof_to_cells = defaultdict(lambda: set())
    for cell in dolfin.cells(V.mesh()):
        local_cell_idx = cell.index()  # MPI-local cell index
        local_dofs = dofmap.cell_dofs(local_cell_idx)  # MPI-local DOF numbers
        # https://fenicsproject.discourse.group/t/indices-of-cells-and-facets-in-parallel/6212/2
        global_cell_idx = cell.global_index()
        global_dofs = l2g[local_dofs]
        for global_dof in global_dofs:
            dof_to_cells[global_dof].add(global_cell_idx)
    return {k: np.array(list(v), dtype="intc") for k, v in dof_to_cells.items()}

def all_patches(V: dolfin.FunctionSpace) -> typing.Dict[int, np.array]:
    """Return a dict of rank-1 np.arrays, mapping DOF number to connected cell numbers.

    Like `my_patches`, but combining data from all MPI processes.
    Each process gets a full copy of all data.
    """
    dof_to_cells = my_patches(V)
    dof_to_cells = dolfin.MPI.comm_world.allgather(dof_to_cells)

    def merge(mappings):
        merged = {}
        while mappings:
            mapping = mappings.pop()
            for global_dof, cell_indices in mapping.items():
                if global_dof not in merged:  # vertex not seen yet?
                    merged[global_dof] = cell_indices
                else:  # vertex seen, add the cells from this MPI process
                    combined = set(merged[global_dof]).union(set(cell_indices))
                    merged[global_dof] = np.array(list(combined),
                                                  dtype="intc")
        return merged
    dof_to_cells = merge(dof_to_cells)
    assert len(dof_to_cells) == V.dim()

    return dof_to_cells

# TODO: generalize for VectorFunctionSpace, TensorFunctionSpace
def map_dG0(V: dolfin.FunctionSpace,
            W: dolfin.FunctionSpace) -> typing.Dict[int, np.array]:
    # Get the patches of cells connected to each global DOF of V.
    V_dof_to_cells = all_patches(V)

    # Get the "patches" of the dG0 space W; each DOF of W corresponds to just one cell.
    # We can use this to find the W DOFs for the cells in each patch of V.
    W_dof_to_cells = all_patches(W)

    assert all(len(cell_indices) == 1 for cell_indices in W_dof_to_cells.values())
    W_dof_to_cell = {dof: cell_indices[0]
                     for dof, cell_indices in W_dof_to_cells.items()}
    cell_to_W_dof = {cell_index: dof
                     for dof, cell_index in W_dof_to_cell.items()}

    # Map each global DOF of V to those DOFs of W that contribute to the patch
    # (note the strategy works correctly for P2 and P3, too)
    V_dof_to_W_dofs = {}
    for global_V_dof, cell_indices in V_dof_to_cells.items():
        cell_indices = V_dof_to_cells[global_V_dof]
        V_dof_to_W_dofs[global_V_dof] = np.array([cell_to_W_dof[k]
                                                  for k in cell_indices], dtype="intc")
    return V_dof_to_W_dofs

def patch_average(f: dolfin.Function,
                  W: typing.Optional[dolfin.FunctionSpace] = None,
                  VtoW: typing.Optional[typing.Dict[int, np.array]] = None) -> dolfin.Function:
    """Patch-average the P1 `Function` `f`, returning the result as a new P1 function.

    Useful as a postprocess step in some nonconforming methods (can eliminate some
    checkerboard modes).

    The optional arguments allow skipping the expensive patch extraction and
    DOF mapping step when there is a need to patch-average functions in a loop:

      `W`: the dG0 space associated with `V = f.function_space()`
      `VtoW`: mapping of DOFs of V onto patches of DOFs of W

    If you use either, both must be specified. To use the optional arguments,
    set them up like this::

        V = f.function_space()
        W = dolfin.FunctionSpace(V.mesh(), "DG", 0)
        VtoW = extrafeathers.meshmagic.map_dG0(V, W)
    """
    # TODO: sanity check input
    V = f.function_space()

    # expensive patch extraction and DOF mapping V -> W
    if not (W and VtoW):
        W = dolfin.FunctionSpace(V.mesh(), 'DG', 0)
        VtoW = map_dG0(V, W)

    # interpolate doesn't work in MPI mode (different partitioning of V and W)
    f_dG0: dolfin.Function = dolfin.project(f, W)

    # make a local copy of the whole dG0 DOF vector in all processes
    all_W_dofs = np.array(range(W.dim()), "intc")
    dG0_vec_copy = dolfin.Vector(dolfin.MPI.comm_self)
    f_dG0.vector().gather(dG0_vec_copy, all_W_dofs)

    # compute patch averages for V DOFs owned by this MPI process
    my_V_dofs = V.dofmap().dofs()
    averages = np.empty(len(my_V_dofs), dtype=np.float64)
    for k, global_V_dof in enumerate(my_V_dofs):
        global_W_dofs = VtoW[global_V_dof]
        averages[k] = dG0_vec_copy[global_W_dofs].sum() / len(global_W_dofs)

    f_pavg = dolfin.Function(V)
    f_pavg.vector()[:] = averages  # MPI-enabled, must run in all processes
    return f_pavg
