# -*- coding: utf-8; -*-
"""Common meta-utilities used by `extrafeathers` itself."""

__all__ = ["make_find_fullmesh_cell", "make_find_fullmesh_facet",
           "is_anticlockwise"]

import typing

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

    If you have already created a cell mapper using `make_find_fullmesh_cell` (for the same
    combination of `fullmesh` and `submesh`), you can pass that as the optional callable to
    avoid unnecessarily creating another one.
    """
    if fullmesh is submesh:
        def find_fullmesh_facet(submesh_facet: dolfin.Facet) -> dolfin.Facet:
            return submesh_facet
    else:
        # A facet has zero area measure, so to be geometrically robust, we match a cell.
        find_fullmesh_cell = find_fullmesh_cell or make_find_fullmesh_cell(fullmesh, submesh)
        def find_fullmesh_facet(submesh_facet: dolfin.Facet) -> dolfin.Facet:
            # Get a submesh cell this facet belongs to - if multiple, any one of them is fine.
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
            fullmesh_facets = list(sorted(fullmesh_facets, key=lambda item: item[1]))  # sort by distance, ascending
            fullmesh_facet, ignored_distance = fullmesh_facets[0]
            return fullmesh_facet
    find_fullmesh_facet.__doc__ = """Given a submesh facet, return the corresponding fullmesh facet."""
    return find_fullmesh_facet


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
