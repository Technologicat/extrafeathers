# -*- coding: utf-8; -*-
"""Mesh-handling utilities.

Currently, some of these are useful for MPI parallel solvers, and some for
full nodal resolution export of data on P2 or P3 triangle meshes.
"""

__all__ = ["my_cells", "all_cells", "nodes_to_array", "collapse_node_numbering",
           "quad_to_tri", "renumber_nodes_by_distance",
           "make_mesh",
           "prepare_linear_export", "refine_for_export", "map_coincident",
           "my_patches", "all_patches", "map_dG0", "patch_average"]

from collections import defaultdict, Counter
import typing

import numpy as np
import scipy.spatial.ckdtree

import dolfin

from .common import is_anticlockwise, maps, multiupdate, freeze, prune, all_values_unique
from .meshfunction import cellvolume


def my_cells(V: dolfin.FunctionSpace, *,
             matplotlibize: bool = False,
             refine: bool = False,
             vertices_only: bool = False) -> typing.Tuple[np.array,
                                                          typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {i1: [x1, y1], ...}

    Nodal (Lagrange) elements only: P, Q, DP and DQ families are ok.

    This can be useful for extracting mesh data for e.g. `matplotlib`. For that,
    see also `as_mpl_triangulation`.

    Note that by default, this returns all nodes, not just the mesh vertices.
    For example, for P2 triangles, you'll get also the nodes at the edge midpoints.
    If that's not what you want, see the `refine` and `vertices_only` flags.

    The original global DOF numbering is preserved.

    The DOF ordering in cell data follows FEniCS conventions (see `demo/refelement.py`).

    This only sees the cells assigned to the current MPI process; the complete
    function space is the union of these cell sets from all processes. See
    `all_cells`, which automatically combines the data from all MPI processes.

    `V`:             Scalar function space.
                       - 2D and 3D ok.
                       - `.sub(j)` of a vector/tensor function space ok.

    `matplotlibize`: 2D only; only used if the resulting output is vertex-only
                     (either `V` is degree-1, `refine=True`, or `vertices_only=True`).

                     If `True`, orient cells anticlockwise, as expected by Matplotlib.

                       - Triangles will list their vertices in an anticlockwise order,
                         and can be used with `matplotlib.tri.Triangulation`.

                       - Quadrilaterals will be oriented anticlockwise, but will list
                         their vertices following the FEniCS convention.

                         Orientation is determined by majority voting between all triplets
                         of adjacent vertices, which works also in the case of nonconvex quads.

                         To construct a triangulation for quads, you can postprocess
                         the extracted cells using the function `quad_to_tri`.

                         To use the quads with `matplotlib.collections.PolyCollection`, you
                         will need to manually change the vertex ordering to Matplotlib's.

    `refine`:        2D only.

                     If `True`, and `V` is degree-2 or degree-3, split each original
                     cell into degree-1 cells (four for degree 2, nine for degree 3),
                     thus producing a corresponding mesh with vertex DOFs only.

                     For triangle meshes, the subtriangles are arranged in an
                     aesthetically pleasing pattern, intended for visualizing P2 or P3
                     data, when interpolating that data as P1 on the once-refined mesh.
                     It is particularly useful for full nodal resolution export into a
                     vertex-based format. See `prepare_linear_export` for the high-level
                     function.

    `vertices_only`: 2D only. Only used if `refine=False`.

                     If `True`, extract only the vertex DOFs, ignoring edge and interior DOFs.

                     Covers some niche use cases with degree-2 and higher function spaces.

    Returns `(cells, nodes)`, where:
        - `cells` is a rank-2 `np.array`, with the entries global DOF numbers,
          one cell per row.

        - `nodes` is a `dict`, mapping from global DOF number to global node coordinates.

          In serial mode for a scalar `FunctionSpace`, the keys are `range(V.dim())`.
          In MPI mode, each process gets a different subrange due to MPI partitioning.

          Also, if `V` is a `.sub(j)` of a `VectorFunctionSpace` or `TensorFunctionSpace`
          (or part of a `MixedElement`), the DOF numbers use the *global* DOF numbering
          also in the sense that each vector/tensor component (and `MixedElement` subfield)
          maps to its own set of global DOF numbers.

          If you instead want the DOF numbers relative to the subspace, collapse
          the subspace first before calling `my_cells` on it.
    """
    if V.ufl_element().num_sub_elements() > 1:
        raise ValueError(f"Expected a scalar `FunctionSpace`, got a function space on {V.ufl_element()}")

    input_degree = V.ufl_element().degree()
    cell_kind = str(V.ufl_element().cell())
    if V.mesh().topology().dim() != 2:
        matplotlibize = False
    if input_degree > 1 and not refine and not vertices_only:
        matplotlibize = False  # TODO: higher-degree orientation flips

    # For now, we support refining in 2D, for P2, DP2, P3, DP3, Q2, DQ2, Q3, and DQ3.
    # TODO: refine also P2/P3 tetras and hexas in 3D? (to allow 3D `refine_for_export`, `prepare_linear_export`)
    if not (V.mesh().topology().dim() == 2 and input_degree > 1):
        refine = False

    # "my" = local to this MPI process
    all_my_cells = []  # e.g. [[i1, i2, i3], ...] in global DOF numbering
    all_my_nodes = {}  # dict to auto-eliminate duplicates (same global DOF)
    dofmap = V.dofmap()
    element = V.element()
    l2g = dofmap.tabulate_local_to_global_dofs()
    for cell in dolfin.cells(V.mesh()):
        local_dofs = dofmap.cell_dofs(cell.index())  # DOF numbers, local to this MPI process
        nodes = element.tabulate_dof_coordinates(cell)  # [[x1, y1], [x2, y2], [x3, y3]], global coordinates
        if not refine:
            if not vertices_only:  # general case: extract all DOFs as-is
                local_dofss = [local_dofs]
                nodess = [nodes]
            else:  # extract vertex DOFs only (ignoring edge and interior DOFs)
                if cell_kind == "triangle":
                    if input_degree == 0:
                        raise ValueError("A degree-0 space has no vertex DOFs")
                    else:  # FEniCS P1/DP1/P2/DP2/P3/DP3 -> FEniCS P1/DP1
                        local_dofss = [local_dofs[:3]]
                        nodess = [nodes[:3]]
                elif cell_kind == "quadrilateral":
                    if input_degree == 0:
                        raise ValueError("A degree-0 space has no vertex DOFs")
                    elif input_degree == 1:  # FEniCS Q1/DQ1 as-is
                        local_dofss = [local_dofs]
                        nodess = [nodes]
                    elif input_degree == 2:  # FEniCS Q2/DQ2 -> FEniCS Q1/DQ1
                        local_dofss = [[local_dofs[0], local_dofs[1], local_dofs[3], local_dofs[4]]]
                        nodess = [[nodes[0], nodes[1], nodes[3], nodes[4]]]
                    elif input_degree == 3:  # FEniCS Q3/DQ3 -> FEniCS Q1/DQ1
                        local_dofss = [[local_dofs[0], local_dofs[1], local_dofs[4], local_dofs[5]]]
                        nodess = [[nodes[0], nodes[1], nodes[4], nodes[5]]]
                    else:
                        raise NotImplementedError(f"{cell_kind} {input_degree}")
                else:
                    raise NotImplementedError(f"{cell_kind} {input_degree}")
        elif input_degree == 2 and cell_kind == "triangle":  # 2D P2 -> once-refined 2D P1
            # Split into four P1 triangles.
            #
            # - DOFs 0, 1, 2 are at the vertices
            # - DOF 3 is on the side opposite to 0
            # - DOF 4 is on the side opposite to 1
            # - DOF 5 is on the side opposite to 2
            #
            # ASCII diagram (showing the DOF numbering on the reference element):
            #
            #     2          2
            #    /|         /|  (1 triangle)
            #   4 3  -->   4-3
            #  /  |       /|/|  (3 triangles)
            # 0-5-1      0-5-1
            #  P2         4×P1
            #
            # See `demo.refelement`, and:
            # https://fenicsproject.discourse.group/t/how-to-get-global-to-local-edge-dof-mapping-for-triangular-p2-elements/5197
            assert len(local_dofs) == 6, len(local_dofs)
            local_dofss = []
            nodess = []
            for i, j, k in ((0, 5, 4), (5, 3, 4), (5, 1, 3),
                            (4, 3, 2)):
                local_dofss.append([local_dofs[i], local_dofs[j], local_dofs[k]])
                nodess.append([nodes[i], nodes[j], nodes[k]])
        elif input_degree == 2 and cell_kind == "quadrilateral":  # 2D Q2 -> once-refined 2D Q1
            # Split into four Q1 quadrilaterals.
            #
            # ASCII diagram:
            #
            # 3-5-4       3-5-4
            # |   |       | | |
            # 6 8 7  -->  6-8-7
            # |   |       | | |
            # 0-2-1       0-2-1
            #  Q2          4×Q1
            #
            # See `demo.refelement`.
            assert len(local_dofs) == 9, len(local_dofs)
            local_dofss = []
            nodess = []
            for i, j, k, ell in ((0, 2, 6, 8), (2, 1, 8, 7),
                                 (6, 8, 3, 5), (8, 7, 5, 4)):
                local_dofss.append([local_dofs[i], local_dofs[j], local_dofs[k], local_dofs[ell]])
                nodess.append([nodes[i], nodes[j], nodes[k], nodes[ell]])
        elif input_degree == 3 and cell_kind == "triangle":  # 2D P3 -> once-refined 2D P1
            # Split into nine P1 triangles.
            #
            # - DOFs 0, 1, 2 are at the vertices
            # - DOFs 3, 4 are on the side opposite to 0
            # - DOFs 5, 6 are on the side opposite to 1
            # - DOFs 7, 8 are on the side opposite to 2
            # - DOF 8 is at the center of the triangle
            #
            # ASCII diagram:
            #
            #       2           2
            #      /|          /|  (1 triangle)
            #     6 4         6-4
            #    /  |  -->   /|/|  (3 triangles)
            #   5 9 3       5-9-3
            #  /    |      /|/|/|  (5 triangles)
            # 0-7-8-1     0-7-8-1
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
        elif input_degree == 3 and cell_kind == "quadrilateral":  # 2D Q3 -> once-refined 2D Q1
            # Split into nine Q1 quadrilaterals.
            #
            # ASCII diagram:
            #
            #  4--6--7--5        4--6--7--5
            #  |        |        |  |  |  |
            # 12 14 15 13       12-14-15-13
            #  |        |  -->   |  |  |  |
            #  8 10 11  9        8-10-11--9
            #  |        |        |  |  |  |
            #  0--2--3--1        0--2--3--1
            #
            # See `demo.refelement`.
            assert len(local_dofs) == 16, len(local_dofs)
            local_dofss = []
            nodess = []
            for i, j, k, ell in ((0, 2, 8, 10), (2, 3, 10, 11), (3, 1, 11, 9),
                                 (8, 10, 12, 14), (10, 11, 14, 15), (11, 9, 15, 13),
                                 (12, 14, 4, 6), (14, 15, 6, 7), (15, 13, 7, 5)):
                local_dofss.append([local_dofs[i], local_dofs[j], local_dofs[k], local_dofs[ell]])
                nodess.append([nodes[i], nodes[j], nodes[k], nodes[ell]])

        # Convert the constituent cells to global DOF numbering,
        # and fix orientation if needed.
        for local_dofs, nodes in zip(local_dofss, nodess):
            # Matplotlib wants anticlockwise ordering when building a Triangulation.
            if matplotlibize:
                if cell_kind == "triangle":
                    if not is_anticlockwise(nodes):
                        local_dofs = local_dofs[::-1]
                        nodes = nodes[::-1]
                else:  # cell_kind == "quadrilateral":
                    # Note that for a **convex** quad, it is sufficient to check
                    # any three consecutive vertices to determine the orientation.
                    #
                    # In the nonconvex case, one of the consecutive vertex triples
                    # will seem to have the opposite orientation. We orient by
                    # majority, which should always do the right thing (allowing
                    # the user to split these quads to triangles later, without
                    # re-orienting).
                    #
                    # Note "consecutive" means walking around the perimeter.
                    # In FEniCS, the numbering is:
                    #
                    #   2-3
                    #   | |
                    #   0-1
                    #   Q1
                    parities = Counter()
                    for subset in ((0, 1, 3), (1, 3, 2), (3, 2, 0), (2, 0, 1)):
                        subset_nodes = [nodes[j] for j in subset]
                        parities[is_anticlockwise(subset_nodes)] += 1
                    assert None not in parities  # no degenerate node triples
                    if parities[False] > parities[True]:  # clockwise majority
                        flipper = [0, 2, 1, 3]  # change orientation, keep FEniCS numbering convention
                        local_dofs = [local_dofs[k] for k in flipper]
                        nodes = [nodes[k] for k in flipper]

            global_dofs = l2g[local_dofs]  # [i1, i2, i3] in MPI-global numbering
            global_nodes = {dof: node for dof, node in zip(global_dofs, nodes)}  # global dof -> coordinates

            all_my_cells.append(global_dofs)
            all_my_nodes.update(global_nodes)
    return all_my_cells, all_my_nodes


def all_cells(V: dolfin.FunctionSpace, *,
              matplotlibize: bool = False,
              refine: bool = False,
              vertices_only: bool = False) -> typing.Tuple[np.array,
                                                           typing.Dict[int, typing.List[float]]]:
    """FunctionSpace -> cell connectivity [[i1, i2, i3], ...], node coordinates {i1: [x1, y1], ...}

    Like `my_cells` (which see for details), but combining data from all MPI processes.
    Each process gets a full copy of all data.
    """
    # Assemble the complete mesh from the partitioned pieces. This treats arbitrary
    # domain shapes correctly. We get the list of triangles from each MPI process
    # and then combine the lists in the root process.
    cells, nodes = my_cells(V, matplotlibize=matplotlibize, refine=refine, vertices_only=vertices_only)
    cellss = dolfin.MPI.comm_world.allgather(cells)
    nodess = dolfin.MPI.comm_world.allgather(nodes)

    # Combine the cell connectivity lists from all MPI processes.
    # The result is a single rank-2 array, with each row the global DOF numbers for a cell, e.g.:
    #   in:  [[i1, i2, i3], ...], [[j1, j2, j3], ...], ...
    #   out: [[i1, i2, i3], ..., [j1, j2, j3], ...]
    # Drop empty lists when combining (in case some MPI process doesn't have any cells)
    our_cells = np.concatenate([lst for lst in cellss if lst])

    # Combine the global DOF index to DOF coordinates mappings from all MPI processes.
    # After this step, each global DOF should have a corresponding vertex.
    def merge(mappings):
        merged = {}
        while mappings:
            merged.update(mappings.pop())
        return merged
    our_nodes = merge(nodess)
    if not vertices_only:
        assert len(our_nodes) == V.dim()  # each DOF of V has coordinates

    return our_cells, our_nodes


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


def collapse_node_numbering(cells: np.array, dofs: np.array) -> np.array:
    """In `cells`, map `dofs[k]` into `k`. Return the remapped cell array.

    The parameters `cells` and `dofs` are as defined in the code snippet below.

    **NOTE**:

    If your data comes from a subspace, prefer its `collapse` method (before
    extracting any cells); it does the same thing.

    This function is mainly useful in niche use cases where you already have
    non-collapsed subspace data in the `all_cells` format, and *then* a need
    arises to collapse the DOF numbering.

    **Motivation**:

    Consider this::

        cells, nodes_dict = meshmagic.all_cells(V)
        dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)

    The array `cells` refers to the nodes using the global DOF numbers `dofs[k]`.
    This is fine as long as we use `nodes_dict`, which maps DOF number to node
    coordinates.

    However, `nodes_array` is just an array, so it uses a zero-based integer index.
    Also, it may be shorter than `max(dofs) - 1`, if there are gaps in the numbering.

    This function converts `cells` from the `dofs[k]` format to the `k` format,
    thus eliminating a layer of indirection.

    **Notes**:

    The cells for each MPI-local mesh part use the global DOF numbers, but refer
    also to unowned nodes, which will not be in the MPI-local `nodes_dict` (as returned
    by `meshmagic.my_cells`). Thus one needs a full (MPI-global) copy of `nodes_dict`
    (as returned by `meshmagic.all_cells`) to interpret even an MPI-local `cells`.

    If `V` is a true scalar `FunctionSpace`, then `dofs` is just the identity mapping,
    and there is no need to collapse the DOFs. But in general:

      - When `V` is a subspace (without collapsing), only a (possibly non-contiguous)
        subset of the global DOFs belongs to the subspace.

        The fully general case is when `V` is on a `MixedElement` with several
        vector/tensor field, and the quantity of interest lives on `V.sub(j).sub(k)`
        (field `j`, component `k`).

      - If you use `quad_to_tri` in MPI-local mode, there will be a gap in the node
        numbering even for true scalar `V`, because the numbers of the added DOFs
        are (as usual for any global DOFs) unique across MPI processes.

        (I.e. the DOF numbers in the MPI-local parts match those of the full mesh.)

    The array `dofs` is already sorted by global DOF number, by `nodes_to_array`.
    Therefore, in the collapsed numbering, the row number `k` of `nodes_array`
    corresponds also to the index of the subspace-relevant slice of the DOF vector.
    This makes plotting easier.

    The subspace-relevant slice is essentially the concatenation of the dofmaps of `V`
    across the MPI processes; see `subspace_dofs` in `mpiplot_prepare`.
    """
    dof_to_row = {dof: k for k, dof in enumerate(dofs)}
    new_cells = [[dof_to_row[dof] for dof in cell] for cell in cells]
    new_cells = np.array(new_cells, dtype=np.int64)
    return new_cells


def quad_to_tri(cells: np.array,
                nodes: typing.Dict[int, typing.List[float]],
                mpi_global: bool) -> typing.Tuple[np.array,
                                                  typing.Dict[int,
                                                              typing.List[float]]]:
    r"""Split quadrilaterals into triangles.

    2D only.

    This function splits each quadrilateral cell into four triangle cells with a
    cross-diagonal pattern::

        +---+       +---+
        |   |       |\ /|
        |   |  -->  | X |
        |   |       |/ \|
        +---+       +---+

    where the `X` denotes the added node.

    If the input cells are Q2, DQ2, Q3, or DQ3, they are stripped to Q1 before
    processing; i.e. we consider only the vertices.

    The main application is accurate visualization of FEM functions on quadrilateral
    meshes (using tooling that only accepts triangle meshes), because the cross-diagonal
    mesh has no preferred diagonal. It needs four times as many cells, and asymptotically
    twice as many DOFs as the original, but that doesn't matter, since it won't be
    used for computation.

    `cells`, `nodes`: As returned by `all_cells` or `my_cells`. Must be quadrilateral,
                      and in FEniCS format (not matplotlib).

                      If your data comes from a subspace, collapse before extracting
                      the cells.

                      This is important, because this function needs zero-based
                      consecutive global DOF numbers to be able to predictably add
                      unique new DOFs across MPI processes.

                      The `nodes` dictionary should always be produced by `all_cells`,
                      because the MPI-local cells refer also to unowned nodes (which are
                      not in the MPI-local `nodes` dictionary).

    `mpi_global`: Whether the input is MPI-global (`all_cells`)
                  or MPI-local (`my_cells`).

    The return value is `(new_cells, new_nodes)`, in the same format as the input.

    "Same format" implies that if `mpi_global=False`, the output will likewise be
    the MPI-local mesh part. However, `new_nodes` will contain also the unowned nodes
    referred to by `cells`, because we need this information anyway to create the new
    nodes.

    If `mpi_global=True`, the output will likewise be MPI-global. Each process gets
    a full copy.

    **CAUTION**:

    In MPI mode, in some versions of FEniCS, trying to `make_mesh` from the data returned
    by this function may crash SCOTCH with an "error in parallel mapping strategy"
    followed by a segfault when FEniCS tries to MPI-distribute the created mesh.
    Renumbering the nodes by proximity (before sending them to `make_mesh`; see
    `renumber_nodes_by_distance`) does not help.

    This appears to be a bug in SCOTCH. In serial mode, the mesh can be created
    just fine.

    As a workaround: this splitter itself works fine. So don't create a `dolfin.Mesh`
    out of the result, but a `matplotlib.tri.Triangulation`, and use it with the
    Matplotlib plotting functions (e.g. `triplot`, `tricontourf`, `tripcolor`).

    This approach requires some more work in getting a DOF vector for function values.
    The original DOF vector data contains the quad vertex values. The quad midpoint
    values can be extracted by a projection (or interpolation) onto a dG0 space on
    the original quad mesh. Then, using the fact that this function places the added
    DOFs at the end, just concatenate the two DOF vectors.
    """
    if len(cells[0]) not in (4, 9, 16):  # Q1/DQ1, Q2/DQ2, Q3/DQ3
        raise ValueError(f"Expected a quadrilateral mesh; got cells with {len(cells[0])} nodes")

    first_node = next(iter(nodes.values()))
    geom_dim = len(first_node)
    if geom_dim != 2:
        raise NotImplementedError(f"This function supports only 2D meshes in 2D space; got geometric dimension {geom_dim}.")

    if set(nodes.keys()) != set(range(len(nodes))):
        raise ValueError("Expected zero-based consecutive DOF numbers. If you want to `quad_to_tri` a subspace, `.collapse()` it before extracting the cells. Also make sure to provide the full `nodes` dictionary (from `all_cells`), not just an MPI-local part (from `my_cells`), even if the cell data comes from `my_cells`.")

    # Take only the Q1/DQ1 part (vertices) if the cells are degree 2 or higher.
    if len(cells[0]) > 4:
        # Convert FEniCS Q2/DQ2/Q3/DQ3 into FEniCS Q1/DQ1.
        # See `plotmagic.as_mpl_triangulation` for pictures.
        if len(cells[0]) == 9:
            cells = [[cell[0], cell[1], cell[3], cell[4]] for cell in cells]
        else:  # 16
            cells = [[cell[0], cell[1], cell[4], cell[5]] for cell in cells]

    # "my" = local to this MPI process.
    num_dofs = len(nodes.keys())
    if mpi_global:
        # Each process processes the full data identically.
        num_quads = len(cells)
        my_first_new_dof = num_dofs
    else:
        # Each process processes its MPI-local part.
        num_my_quads = len(cells)
        num_my_quadss = dolfin.MPI.comm_world.allgather(num_my_quads)
        num_quads = sum(num_my_quadss)

        # The generated DOF numbers will be unique across processes
        # when the original DOFs are in a zero-based consecutive range.
        new_dof_start_offsets = np.concatenate([[0], np.cumsum(num_my_quadss)])
        my_first_new_dof = num_dofs + new_dof_start_offsets[dolfin.MPI.comm_world.rank]

    my_triangles = []  # e.g. [[i1, i2, i3], ...] in global DOF numbering
    my_nodes = {}  # dict to auto-eliminate duplicates (same global DOF)
    for new_dof, cell in enumerate(cells, start=my_first_new_dof):  # MPI-local part
        assert len(cell) == 4  # quadrilateral cells in input

        # Split each Q1 quadrilateral into four P1 triangles.
        # Orientation remains the same as for the original quad.
        #
        # ASCII diagram:
        #
        #   2---3       2---3
        #   |   |       |\ /|
        #   |   |  -->  | X |
        #   |   |       |/ \|
        #   0---1       0---1
        #     Q1         4×P1
        #
        triangles = [[cell[0], cell[1], new_dof],  # bottom
                     [cell[1], cell[3], new_dof],  # right
                     [cell[3], cell[2], new_dof],  # top
                     [cell[2], cell[0], new_dof]]  # left
        my_triangles.extend(triangles)

        # Copy existing nodes
        my_nodes.update({dof: nodes[dof] for dof in cell})  # global dof -> coordinates

        # Add cell midpoint node
        midx = sum(nodes[dof][0] for dof in cell) / len(cell)
        midy = sum(nodes[dof][1] for dof in cell) / len(cell)
        new_node = [midx, midy]
        my_nodes[new_dof] = new_node

    # Postconditions
    if mpi_global:
        assert len(my_triangles) == 4 * num_quads  # each input quad has become four triangles
    else:
        assert len(my_triangles) == 4 * num_my_quads  # each input quad has become four triangles

    return my_triangles, my_nodes


def renumber_nodes_by_distance(cells, nodes, *, origin=None):
    """Sort nodes by distance from a given point, and renumber DOFs to match.

    2D meshes only.

    `cells`, `nodes`: as returned by `all_cells`.
    `origin`: The reference point.
              Can be a length-2 iterable, e.g. `(0.0, 0.0)`.
              If `None`, will autodetect `(min(x[0]), min(x[1]))`.

    The value is the tuple::
        `(cells, nodes, dofs_mapped_to_orig, dofs_orig_to_mapped)`
    where
        `cells` are the input cells represented using the new DOF numbering.
        `nodes` are the sorted nodes in the same format as the input `nodes`,
                using the new DOF numbering: `{new_dof: node_coordinates, ...}`.
        `dofs_mapped_to_orig`: `{new_dof: old_dof, ...}`
        `dofs_orig_to_mapped`: `{old_dof: new_dof, ...}`

    The new DOF numbers are always in a zero-based consecutive range.
    The old DOF numbers are the keys of the input `nodes` dictionary.
    """
    # Collapse the DOF numbers to remove one layer of indirection (`dofs[k]` vs. `k`).
    dofs, nodes_array = nodes_to_array(nodes)
    cells = collapse_node_numbering(cells, dofs)

    if origin is not None:
        x0, y0 = origin
    else:
        x0 = np.min(nodes_array[:, 0])
        y0 = np.min(nodes_array[:, 1])

    # Rank the nodes by (squared) distance from [x0, y0],
    # producing a permutation of the node indices:
    d = nodes_array - np.array([x0, y0])
    dsq = np.sum(d**2, axis=1)
    perm = np.argsort(dsq)

    # Then produce the inverse permutation by the classic O(n) indexing trick:
    #     https://arogozhnikov.github.io/2015/09/29/NumpyTipsAndTricks1.html
    #     https://discuss.codechef.com/t/ambiguous-permutations-explain-the-statement/2430/2
    invperm = np.empty(len(perm), dtype=np.int64)
    invperm[perm] = np.arange(len(perm), dtype=np.int64)

    # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
    #     print(f"d²:      {dsq}")
    #     print(f"perm:    {perm}")
    #     print(f"node:    {np.array(range(len(perm)))}")
    #     print(f"invperm: {invperm}")

    # Now we can use the permutation to renumber the nodes by distance from [x0, y0].
    #
    # For example, consider this data:
    #   d²:      [1.    0.25  0.    0.5   1.    0.25  1.25  1.25  2.    0.625 0.125 0.625 1.125]
    #
    #   perm:    [ 2 10  1  5  3  9 11  0  4 12  6  7  8]   (rank by distance from [x0, y0])
    #   node:    [ 0  1  2  3  4  5  6  7  8  9 10 11 12]   (original index)
    #   invperm: [ 7  2  0  4  8  3 10 11 12  5  1  6  9]
    #
    # In the example data, the node closest to `[x0, y0]` is the original node `2`.
    #
    # So any cell that referred to node `2`, should be changed to refer to node `0`
    # in the new numbering. This is `invperm[2]`, or in general,
    # `invperm[original_node_number]`.
    #
    # In the node array, on the other hand, row `0` of the new array should get its
    # data from row `2` of the original array. This is `perm[0]`, or in general,
    # `perm[row_number]`.

    # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
    #     print(f"Triangles before renumber: {our_triangles}")

    cells = [[invperm[dof] for dof in triangle] for triangle in cells]
    nodes_array[:] = nodes_array[perm, :]

    # Return also a DOF mapping between the original and the collapsed,
    # reordered representation.
    #
    # Continuing the example, the new node `0` is the old node `2`, so the new
    # `dofs[0]` needs to be the old `dofs[2]`, i.e. `dofs[perm[0]]`.
    #
    # Here the array index is the mapped DOF (which are in a zero-based consecutive
    # range), and the value is the original DOF.
    permuted_dofs = dofs[perm]
    dofs_mapped_to_orig = {mapped: orig for mapped, orig in enumerate(permuted_dofs)}
    dofs_orig_to_mapped = {orig: mapped for mapped, orig in dofs_mapped_to_orig.items()}

    # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
    #     print(f"Triangles after renumber:  {our_triangles}")

    d = nodes_array - np.array([x0, y0])
    dsq = np.sum(d**2, axis=1)
    assert ((dsq[1:] - dsq[:-1]) >= 0).all()  # dsq is now non-decreasing
    # print(f"d² after renumber: {dsq}")

    # Pack the node data into the same dict format as the input
    dofs = np.arange(len(nodes_array), dtype=np.int64)  # identity in new numbering
    nodes = {dof: node for dof, node in zip(dofs, nodes_array)}

    return cells, nodes, dofs_mapped_to_orig, dofs_orig_to_mapped


def make_mesh(cells: typing.List[typing.List[int]],
              dofs: np.array,
              vertices: np.array, *,
              distributed: bool = True) -> dolfin.Mesh:
    """Given lists of cells and vertices, build a `dolfin.Mesh`.

    `cells`:       Cell connectivity data as rank-2 `np.array`, one cell per row.
                   (This is the format returned by `all_cells`.)
    `dofs`:        Vertex numbers used by `cells` to refer to `vertices`.
    `vertices`:    Vertex coordinates as rank-2 `np.array`, one vertex per row.

                   `vertices[k]` are the coordinates for global vertex number
                   `dofs[k]`, so that `zip(dofs, vertices)` is the sequence
                   `((vertex_number, coordinates), ...)`.

                   If you're constructing a mesh from scratch, you can use
                   `dofs=range(len(vertices))`. This reason extra level of
                   indirection exists at all is that analyzing an existing
                   mesh will produce data in this format.

                   To create a new mesh based on an existing one, you can get
                   the `dofs` and `vertices` arrays of the original mesh by
                   `dofs, vertices = nodes_to_array(nodes_dict)`,
                   where `nodes_dict` is the dictionary part of the
                   return value of `all_cells` (for a degree-1 `FunctionSpace`
                   based on that mesh).

    `distributed`:  Used when running in MPI mode:

                    If `True`, the mesh will be distributed between the MPI processes.
                    If `False`, each process constructs its own mesh independently.

    Returns a `dolfin.Mesh`.
    """
    # dolfin.MeshEditor is the right tool for the job.
    #     https://fenicsproject.org/qa/12253/integration-on-predefined-grid/
    #     https://fenicsproject.org/olddocs/dolfin/latest/python/_autogenerated/dolfin.cpp.mesh.html#dolfin.cpp.mesh.MeshEditor
    #     https://fenicsproject.org/olddocs/dolfin/latest/cpp/d7/db9/classdolfin_1_1MeshEditor.html
    #     https://bitbucket.org/fenics-project/dolfin/issues/997/quad-hex-meshes-need-ordering-check
    #
    # Using it in MPI mode needs some care. Build the mesh on the root process,
    # then MPI-partition it using `dolfin.MeshPartitioning`.
    #     https://bitbucket.org/fenics-project/dolfin/issues/403/mesheditor-and-meshinit-not-initializing
    #
    if distributed:
        mesh = dolfin.Mesh()  # mesh distributed to MPI processes
    else:
        mesh = dolfin.Mesh(dolfin.MPI.comm_self)  # mesh local to each MPI process

    if not distributed or dolfin.MPI.comm_world.rank == 0:
        editor = dolfin.MeshEditor()
        geometric_dim = np.shape(vertices)[1]

        vertices_per_cell = len(cells[0])
        if vertices_per_cell == 3:
            topological_dim = 2
            cell_kind = "triangle"
        elif geometric_dim == 2 and vertices_per_cell == 4:
            # TODO: Fix conflict with tetrahedra.
            #
            # If `geometric_dim == 2`, then a cell with 4 vertices is guaranteed
            # to be a quadrilateral, but if `3`, then we can't know, because it
            # is possible to have a topologically-2D mesh embedded in 3D space.
            # Maybe add a parameter?
            topological_dim = 2
            cell_kind = "quadrilateral"
        elif geometric_dim == 3 and vertices_per_cell == 4:
            topological_dim = 3
            cell_kind = "tetrahedron"
        elif vertices_per_cell == 6:
            topological_dim = 3
            cell_kind = "hexahedron"
        else:
            raise NotImplementedError(f"Expected triangles or quadrilaterals, but first cell has {len(cells[0])} vertices")

        # See `dolfin.CellType.Type` for cell kinds.
        editor.open(mesh, cell_kind, topological_dim, geometric_dim)

        # Probably, one of the args is the MPI-local count and the other is the global
        # count, but this is not documented, and there is no implementation on the
        # Python level; need to look at the C++ sources of FEniCS to be sure.
        editor.init_vertices_global(len(vertices), len(vertices))
        editor.init_cells_global(len(cells), len(cells))
        for dof, node in zip(dofs, vertices):
            editor.add_vertex_global(dof, dof, node)  # local_index, global_index, coordinates
        for cell_index, cell in enumerate(cells):
            editor.add_cell(cell_index, cell)  # curiously, there is no add_cell_global.
        editor.close()
        mesh.init()
        # print(f"Cells before mesh.order(): {[[vtx.index() for vtx in dolfin.vertices(cell)] for cell in dolfin.cells(mesh)]}")  # DEBUG
        mesh.order()  # TODO: WTF, the goggles do nothing?
        # print(f"Cells after mesh.order():  {[[vtx.index() for vtx in dolfin.vertices(cell)] for cell in dolfin.cells(mesh)]}")  # DEBUG
        # dolfin.info(mesh)  # DEBUG: distributed mode should have all mesh data on root process at this point

    if distributed:
        dolfin.MeshPartitioning.build_distributed_mesh(mesh)
        # dolfin.info(mesh)  # DEBUG: distributed mode should have distributed data to all processes now

    return mesh


def prepare_linear_export(V: typing.Union[dolfin.FunctionSpace,
                                          dolfin.VectorFunctionSpace,
                                          dolfin.TensorFunctionSpace]) -> typing.Tuple[dolfin.Function,
                                                                                       np.array]:
    """Prepare export at full nodal resolution for a vertex-based format.

    2D meshes only.

    See also `refine_for_export` and `map_coincident`, both of which are used by
    this function.

    `V`: P2, P3, DP2, DP3, Q2, Q3, DQ2, or DQ3 space. Scalar/vector/tensor ok.

    Returns the tuple `(u_export, my_V_dofs)`, where:
        - `u_export` is a `Function` on the once-refined degree-1 space based on `V`.
        - `my_V_dofs` is an `np.array` of global DOF numbers of `V` that correspond
          to the MPI-local part of `u_export`.


    **Usage notes**:

    Given something like this typical snippet::

        import dolfin
        from extrafeathers import meshmagic

        mesh = ...  # some 2D triangle mesh
        V = dolfin.FunctionSpace(mesh, "P", 2)  # or 3
        u = dolfin.Function(V)

        # for export to ParaView
        xdmffile_u = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
        xdmffile_u.parameters["flush_output"] = True
        xdmffile_u.parameters["rewrite_function_mesh"] = False

    we would like to set up a P1 space with nodes coincident with those of `V`::

        aux_mesh = meshmagic.refine_for_export(mesh, p=V.ufl_element().degree())
        V_export = FunctionSpace(aux_mesh, "P", 1)
        u_export = Function(V_export)

    and then in the timestep loop, export `u` on this P1 space::

        u_export.assign(dolfin.interpolate(u, V_export))
        xdmffile_u.write(u_export, t)

    which does work in serial mode.

    In MPI mode, the problem is that the DOFs of `V` and `V_export` partition differently
    (essentially because `aux_mesh` is independent of the original `mesh`), so each
    MPI process has no access to some of the `u` data it needs to construct its part
    of `u_export`.

    This is where this function comes in. Set up the linear representation like this::

        u_export, my_V_dofs = meshmagic.prepare_linear_export(V)
        all_V_dofs = np.array(range(V.dim()), "intc")
        u_copy = dolfin.Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V

    Then in the timestep loop, to export `u` on the linear space::

        u.vector().gather(u_copy, all_V_dofs)  # allgather to `u_copy`
        u_export.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(u_export, t)

    Note that if you need them for other purposes, the linear function space and
    its mesh are available as::

        V_export = u_export.function_space()
        aux_mesh = V_export.mesh()
    """
    cell_kind = V.mesh().cell_name()
    if cell_kind not in ("triangle", "quadrilateral"):
        raise NotImplementedError(f"Expected 'triangle' or 'quadrilateral' cells in mesh, got '{cell_kind}'")

    element = V.ufl_element()
    family = str(element.family())
    degree = element.degree()
    tensor_rank = len(element.value_shape())
    if degree < 2 or family not in ("Lagrange",
                                    "Discontinuous Lagrange",
                                    "Q",
                                    "DQ"):
        raise ValueError(f"Expected `V` to use a P2, P3, DP2, DP3, Q2, Q3, DQ2, or DQ3 element, got '{element.family()}' with degree {element.degree()}")

    # TODO: support `prepare_linear_export` for 3D meshes (needs 3D refine support in `my_cells`)
    aux_mesh = refine_for_export(V.mesh(), p=degree)
    constructors = {0: dolfin.FunctionSpace,
                    1: dolfin.VectorFunctionSpace,
                    2: dolfin.TensorFunctionSpace}
    try:
        # Create the corresponding piecewise linear function space
        W = (constructors[tensor_rank])(aux_mesh, family, 1)
        w = dolfin.Function(W)
    except KeyError as err:
        raise ValueError(f"Don't know what to do with tensors of rank > 2, got {tensor_rank}") from err

    # Map the DOFs between the spaces
    WtoV, VtoW = map_coincident(V, W, validate="invertible")

    assert set(WtoV.keys()) == set(range(len(WtoV)))  # whole space, should have zero-based consecutive DOFs
    sorted_WtoV = sorted(WtoV.items(), key=lambda item: item[0])  # sort by global DOF
    vs = [v for k, v in sorted_WtoV]  # get the V DOF numbers only (W DOFs are now 0, 1, 2, ...)
    WtoV = np.array(vs, dtype=np.int64)  # vs are int, because invertible mapping is single-valued

    my_W_dofs = W.dofmap().dofs()  # MPI-local
    my_V_dofs = WtoV[my_W_dofs]  # MPI-local
    return w, my_V_dofs


def refine_for_export(mesh: dolfin.Mesh,
                      p: int = 2,
                      continuous: bool = True) -> dolfin.Mesh:
    r"""Given `mesh`, return a new mesh for exporting data as linear.

    2D meshes only (triangle or quadrilateral ok).

    Like `dolfin.refine(mesh)` without further options; but on triangles,
    we guarantee an aesthetically optimal fill, designed for visualizing
    higher-degree data, when interpolating that data as linear on the
    once-refined mesh.

    For example, we always refine each P2 triangle using this subtriangle
    configuration::

          +               +
         / \             / \
        /   \    -->    +---+
       /     \         / \ / \
      +-------+       +---+---+

    avoiding configurations such as this:

          +               +
         / \       /     /|\
        /   \    -/>    + | +
       /     \   /     / \|/ \
      +-------+       +---+---+

    See also `prepare_linear_export` for an all-in-one solution.

    `p`: The original polynomial degree of the elements your data lives on.
       `p=2` will generate a degree-1 mesh (P1, Q1, DP1, or DQ1) for exporting
             degree-2 data (P2, Q2, DP2, or DQ2, respectively).
       `p=3` will generate a degree-1 mesh (P1, Q1, DP1, or DQ1) for exporting
             degree-3 data (P3, Q3, DP3, or DQ3, respectively).
    `continuous`: Whether your original elements are C0 continuous:
                  If `True`, generate mesh for exporting `P` or `Q` data.
                  If `False`, generate mesh for exporting `DP` or `DQ` data.
    """
    if p not in (2, 3):
        raise ValueError(f"Expected p = 2 or 3, got {p}.")

    cell_kind = mesh.cell_name()
    if cell_kind == "triangle":
        family = "P" if continuous else "DP"
    elif cell_kind == "quadrilateral":
        family = "Q" if continuous else "DQ"
    else:
        raise NotImplementedError(f"Expected 'triangle' or 'quadrilateral' cells in mesh, got '{cell_kind}'")

    # Define an appropriate scalar function space...
    V = dolfin.FunctionSpace(mesh, family, p)
    # TODO: support refining 3D meshes in `my_cells` to allow `refine_for_export` to work on 3D meshes
    cells, nodes_dict = all_cells(V, refine=True)  # ...so that `all_cells` can do our dirty work.
    dofs, nodes_array = nodes_to_array(nodes_dict)
    return make_mesh(cells, dofs, nodes_array, distributed=True)


def map_coincident(V: typing.Union[dolfin.FunctionSpace,
                                   dolfin.VectorFunctionSpace,
                                   dolfin.TensorFunctionSpace],
                   W: typing.Union[dolfin.FunctionSpace,
                                   dolfin.VectorFunctionSpace,
                                   dolfin.TensorFunctionSpace],
                   validate: typing.Optional[str] = None) -> typing.Tuple[typing.Dict[int, typing.Union[int,
                                                                                                        typing.FrozenSet]],
                                                                          typing.Dict[int,
                                                                                      typing.Union[int,
                                                                                                   typing.FrozenSet]]]:
    r"""Build global DOF map of coincident nodes between spaces `V` and `W`.

    The main use case is to export degree-2 or degree-3 data (`V`) in MPI mode at
    full nodal resolution, as once-refined degree-1 data (`W`). In general, the MPI
    partitionings of `V` and `W` will not match, and thus `interpolate(..., W)`
    will not work, because each process is missing some of the input data needed
    to construct its part of the degree-1 representation. This can be worked around
    by hacking the DOF vectors directly. See example below; see also
    `prepare_linear_export` for an all-in-one solution for this use case.

    `V`, `W`: P, DP, Q, or DQ `FunctionSpace`, `VectorFunctionSpace`,
              or `TensorFunctionSpace` on some mesh.

              `V` and `W` must have the same number of subspaces (vector/tensor
              components), if any.

              **NOTE**:

              If both `V` and `W` are discontinuous spaces, then we assume that each
              cell in the mesh of `W` is contained in exactly one cell of the mesh
              of `V`. The assumption is satisfied if the meshes are the same, or
              if the mesh of `W` was produced by refining the mesh of `V`.

              If `V` is a Q or DQ space, it is allowed for `W` to be the `quad_to_tri`
              of `V`.

              The assumption makes it possible to uniquely identify, in a practically
              useful sense, "the same" DOF in the two spaces, when the DOF is on an
              element edge or at a vertex (where several global DOFs of the same
              subspace share the same geometric location, due to the discontinuous
              nature of the space).

              The identification problem does not arise when at least one of the
              spaces `V` or `W` is continuous:

                - If both are spaces are continuous, then both mappings will be
                  single-valued and injective. If every node has a counterpart
                  in the other space, both mappings will be invertible.

                - If one of the spaces is discontinuous and one is continuous,
                  with DOFs at vertices or edges, then one of the mappings
                  will be multi-valued (the same continuous DOF maps to multiple
                  discontinuous DOFs), and the other one will be non-injective
                  (multiple discontinuous DOFs map to the same continuous DOF).

    `validate`: How to validate the resulting mapping W->V (note direction).

        Main use case is programming by contract; if you know that your
        resulting mapping should satisfy one of the following properties,
        declare it to fail-fast when violated.

        If W->V fails the chosen validation, `RuntimeError` is raised.

        Available validation modes:

        `None`:
            No validation.

        "invertible":
            Every DOF of `W` maps to a distinct DOF of `V`, and vice versa,
            so that a single-valued inverse mapping exists.

                W     V
                0 <-> 0
                1 <-> 1
                2 <-> 2

            Example: `V` is P2, and `W` is once-refined P1.

            **NOTE**:

            If W->V is known to be single-valued, "invertible" is equivalent
            with "injective and onto". If it is not known /a priori/ that W->V
            is single-valued, the "invertible" check is stronger, and equivalent
            with "injective and injective-inverse".

            This example passes both the "injective" and "onto" checks, but
            is not invertible:

                W     V
                0 --> 0
                1 --> 1
                   \> 2
                2 --> 3

            Passing the "invertible" check implies that both W->V and V->W
            are single-valued.

        "injective":
            Every DOF of `W` maps to a distinct DOF (or distinct multiple DOFs)
            of `V`, but there may exist DOFs in `V` that are not the image of
            any DOF in `W`.

                 W     V       W     V
                       0             0
                 0 --> 1       0 --> 1
                       2          \> 2
                 1 --> 3       1 --> 3
                       4             4
              single-valued  multi-valued
              injective      injective

            Example: `V` is the `quad_to_tri` of Q1, and `W` is DQ0 on the original mesh.
                     The cell centers of `W` map to the added nodes of `V`, but the
                     original quad vertices on `V` have no counterpart in `W`.

        "injective-inverse":
            Nonstandard counterpart of "injective", for symmetry.
            The mapping V->W (note direction) is injective.

        "onto":
            Each DOF of `V` is the image of one or more DOFs in `W`.
            There are no restrictions on the source DOFs in `W`.

            Particularly, multiple DOFs on `W` may map to the same DOF on `V`,
            and there may exist DOFs in `W` that do not map to any DOF in `V`:

                W     V
                0 --> 0
                1 -/
                2 --> 1
                3

            Example: `V` is P1, and `W` is DP1 on the same mesh.

        "off-from":
            Nonstandard counterpart of "onto", for symmetry.

            Every DOF of `W` maps to one or more DOFs on `V`.
            There are no restrictions on the destination DOFs in `V`.

            In other words, W->V is a total function (as opposed to a partial function
            in the mathematical sense); i.e. it is defined for all DOFs in `W`.

            https://en.wikipedia.org/wiki/Partial_function

                W     V
                0 --> 0
                   \> 1
                1 --> 2
                      3

            Example: `V` is DP1, and `W` is P1 on the same mesh.

    Returns the tuple `(WtoV, VtoW)`, where:
        - The global W DOF `k` matches the global V DOF `j = WtoV[k]`, and
        - The global V DOF `j` matches the global W DOF `k = VtoW[j]`.

          If at least one entry in `WtoV` is multi-valued, then all entries are represented
          in a multi-valued format, `j = frozenset(j0, ...)`. Similarly for `VtoW`.

          If all entries are single-valued, the `frozenset` layer is dropped automatically
          (separately for `WtoV` and `VtoW`).

          To check multi-valuedness, you can
          `is_multivalued = isinstance(next(iter(WtoV.values())), frozenset)`
          (provided that `WtoV` is non-empty).

    Example::

        import numpy as np
        import dolfin
        from extrafeathers import refine_for_export, map_coincident

        mesh = ...

        xdmffile_u = dolfin.XDMFFile(dolfin.MPI.comm_world, "u.xdmf")
        xdmffile_u.parameters["flush_output"] = True
        xdmffile_u.parameters["rewrite_function_mesh"] = False

        V = dolfin.FunctionSpace(mesh, 'P', 2)
        u = dolfin.Function(V)

        aux_mesh = refine_for_export(mesh, p=2)
        W = dolfin.FunctionSpace(aux_mesh, 'P', 1)
        w = dolfin.Function(W)

        WtoV, VtoW = map_coincident(V, W, validate="invertible")

        assert set(WtoV.keys()) == set(range(len(WtoV)))  # whole space, should have zero-based consecutive DOFs
        sorted_WtoV = sorted(WtoV.items(), key=lambda item: item[0])  # sort by global DOF
        vs = [v for k, v in sorted_WtoV]  # get the V DOF numbers only (W DOFs are now 0, 1, 2, ...)
        WtoV = np.array(vs, dtype=np.int64)  # vs are int, because invertible mapping is single-valued

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
    # so by working one component at a time, we make the DOF/node mapping
    # one-to-one.
    if V.num_sub_spaces() > 1:
        spaces = [(V.sub(j), W.sub(j)) for j in range(V.num_sub_spaces())]
    else:
        spaces = [(V, W)]

    # Note that in MPI mode each process constructs its own full copy independently.
    WtoV = {}
    VtoW = {}
    for subV, subW in spaces:
        familyV = str(subV.ufl_element().family())
        if familyV in ("Lagrange", "Q"):
            continuousV = True
        elif familyV in ("Discontinuous Lagrange", "DQ"):
            continuousV = False
        else:
            raise ValueError(f"Unsupported element family '{familyV}'")

        familyW = str(subW.ufl_element().family())
        if familyW in ("Lagrange", "Q"):
            continuousW = True
        elif familyW in ("Discontinuous Lagrange", "DQ"):
            continuousW = False
        else:
            raise ValueError(f"Unsupported element family '{familyW}'")

        # The MPI partitionings of V and W are in general different (independent meshes),
        # so to be able to do the matching, we must gather the data for the whole meshes.
        cellsV, nodesV_dict = all_cells(subV)
        cellsW, nodesW_dict = all_cells(subW)

        subWtosubV, subVtosubW = _map_coincident(cellsV, nodesV_dict, continuousV,
                                                 cellsW, nodesW_dict, continuousW)
        multiupdate(WtoV, subWtosubV)
        multiupdate(VtoW, subVtosubW)

    # Done, let's make the values immutable (to make them hashable).
    WtoV = freeze(WtoV)
    VtoW = freeze(VtoW)

    # contract: postconditions
    if validate == "invertible":
        if not (len(WtoV.keys()) == W.dim()):
            raise RuntimeError("At least one DOF of `W` does not map to any DOF on `V`.")
        if not (len(VtoW.keys()) == V.dim()):
            raise RuntimeError("At least one DOF of `V` does not map to any DOF on `W`.")
        if not (all_values_unique(WtoV)):
            raise RuntimeError("At least two DOFs in `W` map to the same DOF in `V`.")
        # This symmetric additional condition catches the multi-valued pathological case (see docstring).
        if not (all_values_unique(VtoW)):
            raise RuntimeError("At least two DOFs in `V` map to the same DOF in `W`.")
    elif validate == "injective":
        if not (len(WtoV.keys()) == W.dim()):
            raise RuntimeError("At least one DOF of `W` does not map to any DOF on `V`.")
        if not (all_values_unique(WtoV)):
            raise RuntimeError("At least two DOFs in `W` map to the same DOF in `V`.")
    elif validate == "injective-inverse":
        if not (len(VtoW.keys()) == V.dim()):
            raise RuntimeError("At least one DOF of `V` does not map to any DOF on `W`.")
        if not (all_values_unique(VtoW)):
            raise RuntimeError("At least two DOFs in `V` map to the same DOF in `W`.")
    elif validate == "onto":
        if not (len(VtoW.keys()) == V.dim()):
            raise RuntimeError("At least one DOF of `V` does not map to any DOF on `W`.")
    elif validate == "off-from":
        if not (len(WtoV.keys()) == W.dim()):
            raise RuntimeError("At least one DOF of `W` does not map to any DOF on `V`.")

    return prune(WtoV), prune(VtoW)

def _map_coincident(cellsV: np.array, nodesV_dict: typing.Dict[int, typing.List[float]], continuousV: bool,
                    cellsW: np.array, nodesW_dict: typing.Dict[int, typing.List[float]], continuousW: bool):
    """Low-level implementation for `map_coincident`, working with data in the `all_cells` format.

    Must also know whether the spaces `W` and `V` are continuous,
    to choose the correct matching algorithm.

    Note if you use data from `my_cells`, the node dictionaries nevertheless need to
    be complete (from `all_cells`), because in MPI mode, in each process, some cells
    refer to unowned nodes, which are not in the nodes dictionary returned by `my_cells`.

    In MPI mode, each process constructs its own copy.

    Return value is `(WtoV, VtoW)`, where both items are multi-valued mappings.
    """
    match_tol = 1e-8
    WtoV = {}
    VtoW = {}

    dofsW, nodesW = nodes_to_array(nodesW_dict)
    dofsV, nodesV = nodes_to_array(nodesV_dict)

    if continuousV:  # `W` can be continuous or discontinuous
        # On a continuous nodal discretization, each global DOF in the same
        # subspace has a unique geometric location.
        #
        # We can use a simple O(n log(n)) global geometric search, similar to what
        # could be done in serial mode using `dolfin.Mesh.bounding_box_tree`.
        #
        # (That function itself works just fine in MPI mode; the problem are the
        #  different MPI partitionings, so each process is missing some of the
        #  data it needs to find the match.)
        #
        # It doesn't matter whether `W` is continuous or not; if not, multiple
        # DOFs of `W` may map to the same DOF of `V`, but the same algorithm works.
        treeV = scipy.spatial.cKDTree(data=nodesV)
        for dofW, nodeW in zip(dofsW, nodesW):
            distance, k = treeV.query(nodeW)
            if distance > match_tol:  # no coincident node on `V`
                continue
            dofV, nodeV = dofsV[k], nodesV[k]  # noqa: F841, don't need nodeV; just for documentation
            maps(WtoV, dofW, dofV)
            maps(VtoW, dofV, dofW)

    elif continuousW:  # but discontinuous `V`
        # Each DOF on `W` may map to multiple DOFs on `V`.
        #
        # The reliable thing to do here is to run the above algorithm
        # with the roles of `V` and `W` swapped.
        treeW = scipy.spatial.cKDTree(data=nodesW)
        for dofV, nodeV in zip(dofsV, nodesV):
            distance, k = treeW.query(nodeV)
            if distance > match_tol:  # no coincident node on `W`
                continue
            dofW, nodeW = dofsW[k], nodesW[k]  # noqa: F841, don't need nodeW; just for documentation
            maps(VtoW, dofV, dofW)
            maps(WtoV, dofW, dofV)

    else:  # both `W` and `V` discontinuous
        # On a discontinuous nodal discretization, distinct global DOFs on the
        # element boundaries (and at vertices) share the same geometric location.
        #
        # Mapping all of these DOFs to all of the coincident DOFs on the other
        # space is useless in practice. Instead, we assume that each cell of `W`
        # is contained in a single cell of `V` (such as if `W` was produced by
        # refining the mesh of `V`; or if the meshes are the same). This uniquely
        # identifies "the same" DOF in both spaces.
        #
        # TODO: find a suitable generalization when the element boundaries don't agree.

        # First some preparation:
        #
        # Map V DOF number to V cell:
        #   - Cells are always numbered contiguously from zero (row number of `cells` array).
        #   - `nodes[k]` is the node for DOF `dofs[k]`.
        #   - In a dG space proper, each cell has unique DOFs. But:
        #     - If we are dealing with a `quad_to_tri` of, for example, DQ1, then the
        #       triangles generated from the same original quad will share some original
        #       DOFs, as well as the added DOF.
        #       - NOTE: Due to the containment assumption, the `quad_to_tri` data should be
        #         set as `W`, not as `V`. So this will not affect V cell lookup.
        #     - If we are dealing with a DQ1 produced by vis-refining, for example, DQ3,
        #       then some of the DQ1 quads (those generated from the same original DQ3 quad)
        #       will share some DOFs. If `V` is the refined space, and `W` is `quad_to_tri`
        #       of that, then some `V` DOFs belong to several cells.
        # Thus, to cover the general case, the mapping from V DOF to V cell must be multivalued.
        dof_to_cell_V = {}
        for cellV_idx, cellV in enumerate(cellsV):
            for dofV in cellV:
                maps(dof_to_cell_V, dofV, cellV_idx)

        # Map DOF to row of `nodes` array on both spaces.
        # These mappings are always invertible, because they are just a renumbering of the DOFs.
        dof_to_row_V = {dof: k for k, dof in enumerate(dofsV)}
        dof_to_row_W = {dof: k for k, dof in enumerate(dofsW)}

        # Match one W cell at a time. By assumption, it will be contained in exactly one V cell.
        treeV = scipy.spatial.cKDTree(data=nodesV)
        for cellW in cellsW:
            # Find the correct V cell by voting.
            #
            #  - Distance-search for coincident DOFs on V for **all** DOFs
            #    on the W cell. Count how many matches each candidate V cell gets.
            #  - All DOFs of the W cell should be in one cell on V.
            #  - Any cell that matches only some of the DOFs is a neighbor.
            #
            # See `img/triangles.svg`; consider the DOFs of the shaded triangle.
            #
            # The figure also shows why cell midpoints are useless in finding
            # the corresponding V cell; the midpoint of a neighboring V cell
            # may be closer (to the midpoint of the W cell) than the correct one.
            ballot = Counter()
            for dofW in cellW:
                nodeW = nodesW[dof_to_row_W[dofW]]
                # Theoretically, the correct maximum number of neighbors to look
                # for is infinity, since any number of triangles may meet at
                # a vertex, and in a discontinuous space, each of them will have
                # one unique DOF there.
                #
                # However, for practical FEM meshes, only a few elements will
                # meet at a vertex. We also use the `distance_upper_bound` option
                # to prune the search, since we look for coincident nodes only.
                distances, ks = treeV.query(nodeW, k=10,
                                            distance_upper_bound=1e-8)
                for distance, k in zip(distances, ks):
                    if distance > match_tol:
                        break
                    dofV, nodeV = dofsV[k], nodesV[k]  # noqa: F841, don't need nodeV; just for documentation
                    # Vote for all V cells this V DOF belongs to.
                    for cellV_idx in dof_to_cell_V[dofV]:
                        ballot[cellV_idx] += 1

                # Match the `W` DOF against `V` cell midpoints to handle `quad_to_tri` data correctly.
                # Some adjacent cells may then have an equal number of votes; to decide, we need to
                # check the midpoint.
                for cellV_idx in ballot.keys():
                    # Dirty HACK: to find midpoint, just average all node coordinates,
                    # also for higher-degree cells, since we don't have the cell type info.
                    #
                    # As long as any edge and interior DOFs are placed symmetrically on the
                    # reference element (as they are for P2/P3/Q2/Q3/DP2/DP3/DQ2/DQ3), this
                    # will give the correct result.
                    vtxs = [nodesV[dofV] for dofV in cellsV[cellV_idx]]
                    xmid = sum([x for x, y in vtxs]) / len(vtxs)
                    ymid = sum([y for x, y in vtxs]) / len(vtxs)
                    midpointV = np.array([xmid, ymid])
                    dsq = sum((nodeW - midpointV)**2)
                    if dsq > match_tol:
                        continue
                    ballot[cellV_idx] += 1
            if len(ballot) == 0:  # no cell on `V` had nodes coincident with those of this `W` cell
                continue

            # If at least one coincident node exists, then by the containment assumption,
            # exactly one cell wins the vote.
            # (A failure here has caused some silent bugs during development, so let's check it.)
            two_most_common = ballot.most_common(2)
            if len(two_most_common) == 2:
                [[e1, votes_most_common], [e2, votes_second_most_common]] = two_most_common
                assert votes_most_common > votes_second_most_common  # winner of the V cell vote is unique

            [[cellV_idx, votes]] = ballot.most_common(1)
            # Often, `votes == len(cellW)`, but not always, if not all `W` DOFs have a counterpart on `V`.
            # (E.g. if `W` is the `quad_to_tri` of `V`, the added DOFs have no counterpart.)

            # Now that we know the correct V cell, walk again the DOFs on the W cell,
            # but match only against nodes of the correct cell . There are just a few,
            # and we need to repeat this for each W cell (nothing useful to cache),
            # so `np.argmin` is fine.
            for dofW in cellW:
                nodeW = nodesW[dof_to_row_W[dofW]]

                # Next comes some indexing... take a deep breath:
                #
                # Find the rows of the `nodesV` array for the V DOFs of the
                # vote-winning cell.
                rowsV = np.array([dof_to_row_V[dofV] for dofV in cellsV[cellV_idx]],
                                 dtype=np.int64)
                relevant_nodesV = nodesV[rowsV]

                d = relevant_nodesV - nodeW
                dsq = np.sum(d**2, axis=1)
                r = np.argmin(dsq)  # row of relevant nodes array
                if dsq[r] > match_tol:  # no coincident node on `V`
                    continue
                k = rowsV[r]  # row of complete `nodesV` array

                # ...which gives us the V DOF number:
                dofV, nodeV = dofsV[k], nodesV[k]  # noqa: F841, don't need nodeV; just for documentation

                # ...which finally gets us what we want:
                maps(WtoV, dofW, dofV)
                maps(VtoW, dofV, dofW)

    return WtoV, VtoW


def my_patches(V: dolfin.FunctionSpace) -> typing.Dict[int, np.array]:
    """Map each global DOF number to connected global cell numbers.

    The return value is a dict of rank-1 np.arrays.

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
        # Note this strategy works correctly also when V is a P2 or P3 space;
        # the nodes on facets will have smaller patches (just two cells) than
        # the nodes at vertices.
        for global_dof in global_dofs:
            dof_to_cells[global_dof].add(global_cell_idx)
    return {k: np.array(list(v), dtype="intc") for k, v in dof_to_cells.items()}

def all_patches(V: dolfin.FunctionSpace) -> typing.Dict[int, np.array]:
    """Map each global DOF number to connected global cell numbers.

    The return value is a dict of rank-1 np.arrays.

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
                if global_dof not in merged:  # This global DOF not seen yet
                    merged[global_dof] = cell_indices
                else:  # DOF already seen, add new unique cells from this MPI process
                    combined = set(merged[global_dof]).union(set(cell_indices))
                    merged[global_dof] = np.array(list(combined),
                                                  dtype="intc")
        return merged
    dof_to_cells = merge(dof_to_cells)
    assert len(dof_to_cells) == V.dim()  # each DOF was seen

    return dof_to_cells

def map_dG0(V: typing.Union[dolfin.FunctionSpace,
                            dolfin.VectorFunctionSpace,
                            dolfin.TensorFunctionSpace],
            W: typing.Union[dolfin.FunctionSpace,
                            dolfin.VectorFunctionSpace,
                            dolfin.TensorFunctionSpace]) -> typing.Tuple[typing.Dict[int, np.array],
                                                                         typing.Dict[int, int]]:
    """Map each global DOF of V to DOFs on dG0 space W.

    That is, determine the W DOFs that contribute to each V DOF if we were
    to patch-average a function on V by projecting it to W (to get a
    representative value for each cell) and then averaging the cell values
    appropriately.

    The return value is `(V_dof_to_W_dofs, W_dof_to_cell)`.
    """
    if W.num_sub_spaces() != V.num_sub_spaces():
        raise ValueError(f"V and W must have as many subspaces; V has {V.num_sub_spaces()}, but W has {W.num_sub_spaces()}.")

    # In a vector/tensor function space, each geometric node (global DOF
    # coordinates) has an independent instance for each field component,
    # so by working one component at a time, we make the DOF/node mapping
    # one-to-one.
    if V.num_sub_spaces() > 1:
        spaces = [(V.sub(j), W.sub(j)) for j in range(V.num_sub_spaces())]
    else:
        spaces = [(V, W)]
    V_dof_to_W_dofs = {}
    W_dof_to_cell_all = {}

    # Note that in MPI mode each process constructs its own full copy independently.
    seenV = set()
    seenW = set()
    for subV, subW in spaces:
        if not (str(subW.ufl_element().family()) == "Discontinuous Lagrange" and
                subW.ufl_element().degree() == 0):
            raise ValueError(f"Expected `W` to be a discontinuous Lagrange space with degree 0; got a {subW.ufl_element().family()} space with degree {subW.ufl_element().degree()}")

        # The MPI partitionings of V and W are in general different, so to be
        # able to do the matching, we must gather the data for the whole mesh.

        # Get the patches of cells connected to each global DOF of V.
        # Note that for a vector/tensor space, multiple DOFs will map to the same set of cells.
        V_dof_to_cells = all_patches(subV)

        # Get the "patches" of the dG0 space W; each DOF of W is connected to just one cell.
        # We can use this to find the W DOFs for the cells in each patch of V...
        W_dof_to_cells = all_patches(subW)
        assert all(len(cell_indices) == 1 for cell_indices in W_dof_to_cells.values())

        W_dof_to_cell = {dof: cell_indices[0]
                         for dof, cell_indices in W_dof_to_cells.items()}
        W_dof_to_cell_all.update(W_dof_to_cell)

        # ...by inverting, so that each global cell maps to a DOF of W:
        cell_to_W_dof = {cell_index: dof
                         for dof, cell_index in W_dof_to_cell.items()}
        W_dof_to_cell_all.update(W_dof_to_cell)
        assert len(cell_to_W_dof) == subW.mesh().num_entities_global(subW.mesh().topology().dim())

        # Map each global DOF of V to those DOFs of W that contribute to the patch.
        # The strategy works correctly also when V is a P2 or P3 space, because
        # `my_patches` does.
        for global_V_dof, cell_indices in V_dof_to_cells.items():
            seenV.add(global_V_dof)
            cell_indices = V_dof_to_cells[global_V_dof]
            V_dof_to_W_dofs[global_V_dof] = np.array([cell_to_W_dof[k]
                                                      for k in cell_indices],
                                                     dtype="intc")
            [seenW.add(W_dof) for W_dof in V_dof_to_W_dofs[global_V_dof]]
    # postconditions
    assert set(range(V.dim())) - seenV == set()  # all DOFs of V seen
    assert set(range(W.dim())) - seenW == set()  # all DOFs of W mapped to at least once
    return V_dof_to_W_dofs, W_dof_to_cell_all

# TODO: maybe this should just patch-average a dG0 function onto a target space `V`
def patch_average(f: dolfin.Function,
                  W: typing.Optional[typing.Union[dolfin.FunctionSpace,
                                                  dolfin.VectorFunctionSpace,
                                                  dolfin.TensorFunctionSpace]] = None,
                  VtoW: typing.Optional[typing.Dict[int, np.array]] = None,
                  Wtocell: typing.Optional[typing.Dict[int, int]] = None,
                  cell_volume: typing.Optional[typing.Union[dolfin.Function,
                                                            dolfin.Expression]] = None,
                  *, mode: str = "project") -> dolfin.Function:
    """Patch-average the `Function` `f` into a new function on the same function space.

    Useful as a postprocess step in some nonconforming methods (can eliminate some
    checkerboard modes).

      `f`: a scalar FEM function with nodal DOFs.
      `mode`: how to produce the piecewise constant cell values (which will be averaged
              to compute the patch average). One of "project", "interpolate". These
              correspond to using the DOLFIN function of the same name.

    The optional arguments allow skipping the expensive patch extraction and
    DOF mapping step when there is a need to patch-average functions in a loop.
    They are also mandatory if `f` is a vector or tensor function.

      `W`: The dG0 space associated with `V = f.function_space()`.
      `VtoW`, `Wtocell`: as returned by `map_dG0`
      `cell_volume`: The local cell volume of `W.mesh()`.

    To use the optional arguments, set them up like this (and be sure to
    set up all of them)::

        import dolfin
        from extrafeathers import map_dG0, cellvolume

        V = f.function_space()
        W = dolfin.FunctionSpace(V.mesh(), "DG", 0)
        VtoW, Wtocell = map_dG0(V, W)
        cell_volume = cellvolume(W.mesh())

    Note `W` should be a `FunctionSpace`, `VectorFunctionSpace`, or
    `TensorFunctionSpace`, as appropriate (i.e. the same kind `V` is).


    **CAUTION**:

    Rather than patch-averaging, it is in general better to interpolate to a dG0
    (elementwise constant) space and then project back to the input (continuous)
    space::

        import dolfin

        W = dolfin.FunctionSpace(V.mesh(), "DG", 0)
        f_averaged = dolfin.project(dolfin.interpolate(f, W), V)

    Patch-averaging gives the same result if `V` is a P1 space; in all other cases,
    `project(interpolate(...))` does the same thing in spirit, but correctly.

    This function is provided just because patch-averaging is a classical
    postprocessing method.


    **Algorithm**:

    In "project" mode:

      1) L2-project `f` onto the dG0 space `W`, i.e. solve for `w` such that:
             ∫ v w dΩ = ∫ v f dΩ  ∀ v ∈ W
      2) For each DOF of `V`, average the dG0 cell values over the patch of cells
         connected to that DOF, weighted by the relative cell volume of each
         contributing cell.
      3) Define a new function on `V`, setting the patch averages from step 2
         as the values of its DOFs.

    In "interpolate" mode, step 1 is changed to just sample `f` at the cell midpoints
    to produce the dG0 function.


    **Notes** on "project" mode:

    There is a related discussion in Hughes (sec. 4.4.1) on pressure smoothing.
    The difference is that our `f` is C0 continuous, and the least-squares averaging
    is performed in projection onto dG0. Hughes, on the other hand, starts from a
    piecewise constant (i.e. dG0) function, and performs least-squares averaging in
    projection of that function onto a C0 continuous finite element space.

    Consider representing `w` as a Galerkin series:

        w := ∑k wk φk,

    where wk are the coefficients, and φk ∈ W are the basis functions. Minimizing
    (w - f)² in a least-squares sense means that must find coefficients wk such that:

        0 = ∂/∂wi ∫ (w - f)² dΩ  (∀ i)
          = ∂/∂wi ∫ (∑i wi φi - f) (∑k wk φk - f) dΩ
          = ∫ φi (∑k wk φk - f) dΩ
          = ∑k wk ∫ φi φk dΩ - ∫ φi f dΩ

    (This extremum is guaranteed to be a minimum, because the expression is quadratic
     in w, with a positive leading term.)

    In other words:

        ∑k ∫ φi φk dΩ wk = ∫ φi f dΩ  (∀ i)

    which is exactly the Galerkin discretization of the L2 projection onto `W`.

    References:
        Thomas J. R. Hughes. 2000. The Finite Element Method: Linear Static and Dynamic
        Finite Element Analysis. Dover. Corrected and updated reprint of the 1987 edition.
        ISBN 978-0-486-41181-1.
    """
    if mode not in ("project", "interpolate"):
        raise ValueError(f"Expected `mode` to be one of 'project', 'interpolate'; got {mode}")
    if W:
        if not (str(W.ufl_element().family()) == "Discontinuous Lagrange" and
                W.ufl_element().degree() == 0):
            raise ValueError(f"Expected `W` to be a discontinuous Lagrange space with degree 0; got a {W.ufl_element().family()} space with degree {W.ufl_element().degree()}")
    if any((W, VtoW, Wtocell, cell_volume)) and not all((W, VtoW, Wtocell, cell_volume)):
        raise ValueError(f"When the optional arguments are used, all of them must be provided. Got W = {W}, VtoW = {VtoW}, Wtocell = {Wtocell}, cell_volume = {cell_volume}.")

    V = f.function_space()

    # Patch extraction and DOF mapping V -> W.
    if not (W and VtoW and Wtocell and cell_volume):
        W = dolfin.FunctionSpace(V.mesh(), 'DG', 0)
        VtoW, Wtocell = map_dG0(V, W)
        cell_volume = cellvolume(W.mesh())
    # TODO: API of `map_dG0`? Does a dict make sense at all?
    assert set(Wtocell.keys()) == set(range(W.dim()))
    # Dict[int, int] -> np.array[int], the row index is the key.
    Wtocell = np.array([cell for global_W_dof, cell in sorted(Wtocell.items(), key=lambda item: item[0])])

    P = dolfin.interpolate if mode == "interpolate" else dolfin.project
    f_dG0: dolfin.Function = P(f, W)

    # Make a local copy of the whole dG0 DOF vector in all processes.
    all_W_dofs = np.array(range(W.dim()), "intc")
    dG0_vec_copy = dolfin.Vector(dolfin.MPI.comm_self)
    f_dG0.vector().gather(dG0_vec_copy, all_W_dofs)

    # Also get a full copy of the cell volume vector in all processes.
    all_cell_volumes = np.concatenate(dolfin.MPI.comm_world.allgather(cell_volume.array()))

    # Compute patch averages for V DOFs owned by this MPI process.
    # (The patches may refer to cells not owned by this process.)
    my_V_dofs = V.dofmap().dofs()
    averages = np.empty(len(my_V_dofs), dtype=np.float64)
    for k, global_V_dof in enumerate(my_V_dofs):
        global_W_dofs = VtoW[global_V_dof]
        cells = Wtocell[global_W_dofs]
        patch_cell_volumes = all_cell_volumes[cells]
        patch_total_volume = patch_cell_volumes.sum()
        averages[k] = (dG0_vec_copy[global_W_dofs] * patch_cell_volumes).sum() / patch_total_volume

    f_pavg = dolfin.Function(V)
    f_pavg.vector()[:] = averages  # MPI-enabled, must run in all processes
    return f_pavg
