# -*- coding: utf-8; -*-
"""Common meta-utilities used by `extrafeathers` itself.

These are public, but not imported to the top-level namespace.
"""

__all__ = ["make_find_fullmesh_cell", "make_find_fullmesh_facet",
           "is_anticlockwise",
           "maps", "multiupdate", "freeze", "prune", "all_values_unique", "all_valuesets_unique",
           "minmax"]

import typing

import numpy as np

import dolfin


def make_find_fullmesh_cell(fullmesh: dolfin.Mesh,
                            submesh: typing.Union[dolfin.SubMesh, dolfin.Mesh]) -> typing.Callable:
    """Make a function that maps a cell of `submesh` to the corresponding cell of `fullmesh`."""
    if fullmesh is submesh:
        def find_fullmesh_cell(submesh_cell: dolfin.Cell) -> dolfin.Cell:
            return submesh_cell
    else:
        _bbt = fullmesh.bounding_box_tree()
        def find_fullmesh_cell(submesh_cell: dolfin.Cell) -> dolfin.Cell:
            fullmesh_cell_index = _bbt.compute_first_entity_collision(submesh_cell.midpoint())
            fullmesh_cell = dolfin.Cell(fullmesh, fullmesh_cell_index)
            return fullmesh_cell
    find_fullmesh_cell.__doc__ = """Given a submesh cell, return the corresponding fullmesh cell."""
    return find_fullmesh_cell


def make_find_fullmesh_facet(fullmesh: dolfin.Mesh,
                             submesh: typing.Union[dolfin.SubMesh, dolfin.Mesh],
                             find_fullmesh_cell: typing.Optional[typing.Callable] = None) -> typing.Callable:
    """Make a function that maps a facet of `submesh` to the corresponding facet of `fullmesh`.

    If you have already created a cell mapper using `make_find_fullmesh_cell`
    (for the same combination of `fullmesh` and `submesh`), you can pass that
    as the optional callable to avoid unnecessarily creating another one.
    """
    if fullmesh is submesh:
        def find_fullmesh_facet(submesh_facet: dolfin.Facet) -> dolfin.Facet:
            return submesh_facet
    else:
        # A facet has zero area measure, so to be geometrically robust, we match a cell.
        find_fullmesh_cell = find_fullmesh_cell or make_find_fullmesh_cell(fullmesh, submesh)
        def find_fullmesh_facet(submesh_facet: dolfin.Facet) -> dolfin.Facet:
            # Get a submesh cell this facet belongs to - if multiple, any one
            # of them is fine.
            submesh_cells_for_facet = list(dolfin.cells(submesh_facet))
            assert submesh_cells_for_facet  # there should always be at least one
            submesh_cell = submesh_cells_for_facet[0]

            # Find the corresponding fullmesh cell.
            #
            # This fullmesh cell always belongs to submesh, but other fullmesh
            # cells connected to the facet might not.
            fullmesh_cell = find_fullmesh_cell(submesh_cell)

            # Find the corresponding fullmesh facet in the data for the fullmesh cell.
            # We pick the fullmesh facet whose midpoint has minimal (actually zero)
            # distance to the midpoint of submesh_facet.
            fullmesh_facets = [(fullmesh_facet, fullmesh_facet.midpoint().distance(submesh_facet.midpoint()))
                               for fullmesh_facet in dolfin.facets(fullmesh_cell)]
            # sort by distance, ascending
            fullmesh_facets = list(sorted(fullmesh_facets, key=lambda item: item[1]))
            fullmesh_facet, ignored_distance = fullmesh_facets[0]
            return fullmesh_facet
    find_fullmesh_facet.__doc__ = """Given a submesh facet, return the corresponding fullmesh facet."""
    return find_fullmesh_facet


def is_anticlockwise(ps: typing.List[typing.List[float]]) -> typing.Optional[bool]:
    """[[x1, y1], [x2, y2], [x3, y3]] -> whether the points are listed anticlockwise.

    In the degenerate case where the points are exactly on a line
    (up to machine precision), returns `None`.

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

# --------------------------------------------------------------------------------

def maps(d, k, v):
    """Helper for building multi-valued mappings.

    Add `v` to the value set `d[k]`, creating the set if necessary.

    Pronounced as "d maps k to v" [and possibly to other values, too].
    """
    if k not in d:
        d[k] = set()
    d[k].add(v)

def multiupdate(dst, src):
    """Update a multi-valued mapping `dst` by a multi-valued mapping `src`.

    Copy new keys as-is; for existing keys, merge into the existing valueset.
    """
    for k, v in src.items():
        if k in dst:
            dst[k].update(v)
        else:
            dst[k] = v

def freeze(d):
    """Convert the value sets of a multi-valued mapping into frozensets.

    The use case is to make the value sets hashable once they no longer need to be edited.
    """
    return {k: frozenset(v) for k, v in d.items()}

def prune(d):
    """If possible, make the multi-valued mapping `d` single-valued.

    If all value sets of `d` have exactly one member, remove the set layer,
    and return the single-valued dictionary.

    Else return `d` as-is.
    """
    if all(len(v) == 1 for v in d.values()):
        return {k: next(iter(v)) for k, v in d.items()}
    return d

def all_values_unique(d):
    """Return whether no value appears more than once across all value sets of multi-valued mapping `d`."""
    seen = set()
    for multivalue in d.values():
        if any(v in seen for v in multivalue):
            return False
        seen.update(multivalue)
    return True

def all_valuesets_unique(d):
    """Return whether no two value sets are the same in multi-valued mapping `d`."""
    d = freeze(d)
    seen = set()
    for multivalue in d.values():
        if multivalue in seen:
            return False
        seen.add(multivalue)
    return True

# --------------------------------------------------------------------------------

def minmax(p: dolfin.Function, *,
           mode: str = "raw",
           take_abs: bool = False) -> typing.Union[typing.Tuple[float, float],
                                                   typing.Tuple[typing.List[float],
                                                                typing.List[float]]]:
    """Find min and max of a FEM function.

    Supports MPI, vectors and tensors. Useful for determining a colormap range
    for plotting.

    `p`: A FEM function on some function space or on a subspace.

    `mode`: Minmax what. One of:

            "raw": All DOFs of `p`. Valid for any `p`.

                   (In case `p` is a vector or tensor field, do not care which
                    component/subspace each DOF belongs to.)

        When `p` is a vector or tensor field, additionally available modes:

            "l2": Euclidean length of the vector made out of the components of `p`.
                  `p` must be a vector or tensor field.

            "components": All components (subspaces); return lists as results,
                          one entry per subspace.

    `take_abs`: If `True`, then compute `min(abs(...))` and `max(abs(...))`.
                If `False`, then compute `min(...)` and `max(...)`.

                Ignored when `mode="l2"`.
    """
    assert mode in ("raw", "l2", "components")

    def getdofs(W):
        """Get all DOF values of `p` in space `W`. Account for `W` possibly being a subspace."""
        dofmaps = dolfin.MPI.comm_world.allgather(W.dofmap().dofs())
        subspace_dofs = np.concatenate(dofmaps)
        p_vec_copy = dolfin.Vector(dolfin.MPI.comm_self)  # MPI-local
        p.vector().gather(p_vec_copy, subspace_dofs)  # allgather; takes global DOF indices
        return np.array(p_vec_copy)

    W = p.function_space()

    if mode == "components":  # all components, separately; shorthand for:
        minps, maxps = [], []
        for j in range(W.num_sub_spaces()):
            minp, maxp = minmax(p.sub(j), take_abs=take_abs, mode="raw")
            minps.append(minp)
            maxps.append(maxp)
        return minps, maxps

    if mode == "l2":  # euclidean length
        vecs = [getdofs(W.sub(j)) for j in range(W.num_sub_spaces())]
        vecs = [vec**2 for vec in vecs]  # squared nodal values for each component
        pvec = sum(vecs)**0.5
    else:  # mode == "raw":  all DOFs, don't care about which component
        pvec = getdofs(W)
        if take_abs:
            pvec = np.abs(pvec)
    return min(pvec), max(pvec)
