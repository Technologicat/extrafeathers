# -*- coding: utf-8; -*-
"""Automatic extractor for boundaries between subdomains, for use with marked submeshes.

Loosely based on the following, and then improved:
    https://fenicsproject.org/qa/13344/boundary-conditions-on-subdomain/
    https://fenicsproject.org/qa/397/how-define-dirichlet-boundary-conditions-with-meshfunction/
"""

__all__ = ["find_subdomain_boundaries"]

import typing

import dolfin

from .common import make_find_fullmesh_facet


def find_subdomain_boundaries(fullmesh: dolfin.Mesh,
                              submesh: typing.Union[dolfin.SubMesh, dolfin.Mesh],
                              subdomains: typing.Optional[dolfin.MeshFunction],
                              boundary_spec: typing.Dict[typing.FrozenSet[int], int],
                              callback: typing.Optional[typing.Callable] = None) -> dolfin.MeshFunction:
    """Find and tag domain and subdomain boundaries in a mesh.

    Produces a `MeshFunction` on the facets of `submesh`, which can then be
    used in the solver to set boundary conditions on `submesh`. This is useful
    e.g. for FSI (fluid-structure interaction) problems, but applicable to just
    CFD (computational fluid dynamics), too (by just discarding the mesh for
    the solid subdomain).

    Serial mode only. (As of FEniCS 2019, `SubMesh` is not available in MPI mode anyway.)

    `submesh`: The submesh of `fullmesh` for which to produce the boundary-identifying
               `MeshFunction`.

    `fullmesh`: The full mesh that `submesh` is part of.

                If no subdomains are used, `fullmesh` can be the same as `submesh`.
                This is useful if one wishes to tag the outer boundary only.

    `subdomains`: A `MeshFunction` on cells of `fullmesh`, specifying the subdomains.
                  (See example below.)

                  If no subdomains are used, can be `None`.

    `boundary_spec`: A dictionary that specifies the boundary tag for each desired pair
                     of subdomains. The format is::

                         {frozenset({subdomain_1_tag, subdomain_2_tag}): boundary_tag,
                          ...}

                     The `frozenset(...)` makes the key hashable. Each tag is a
                     nonnegative integer.

                     Several subdomain pairs are allowed to map to the same boundary_tag;
                     just add a separate entry for each subdomain pair.

                     `boundary_spec` can be the empty dict. Then:
                       - If `subdomains is None`, all boundary facets are treated as if
                         they were on the outer boundary.
                       - If `subdomains` is specified, all subdomain boundaries will be
                         ignored.

    `callback`: If specified, it is called for each facet that is on the outer
                boundary (i.e. on the boundary, but no neighboring subdomain
                exists). Facilitates tagging all boundaries of `submesh` in one
                go; the solver then only needs one `MeshFunction` to identify
                all boundaries when setting boundary conditions.

                The call signature of `callback` is::

                         f(submesh_facet: Facet, fullmesh_facet: Facet) -> Optional[int]

                The return value should be a boundary tag or `None`. If a
                boundary tag is returned, the facet is tagged (in the resulting
                `MeshFunction`) by that tag. If `None` is returned, the facet
                will not be tagged.
    """
    facet_dim = submesh.topology().dim() - 1
    boundary_parts = dolfin.MeshFunction('size_t', submesh, facet_dim)  # value_type, mesh, dimension, [default_value]
    boundary_parts.set_all(0)

    find_fullmesh_facet = make_find_fullmesh_facet(fullmesh, submesh)

    # The mesh function for the boundary parts is defined over the submesh only,
    # so it must be indexed with the submesh facet. The easiest way to get the
    # submesh facets is to loop over all of them.
    #
    # However, subdomains are defined over the full mesh, not just a submesh,
    # so the mesh function for the subdomains must be indexed with the
    # fullmesh cell, not the submesh one.
    #
    # So we must match the submesh facet to a fullmesh facet, and then check the
    # subdomain IDs of the fullmesh cells that facet belongs to. In a conforming
    # mesh, this results in exactly one or two subdomains. If there are two, then
    # the facet is on a boundary between subdomains.
    #
    # These considerations lead to the following algorithm:
    for submesh_facet in dolfin.facets(submesh):
        # We only need to consider boundary facets. A facet on the boundary
        # of the submesh is connected to exactly one cell in the submesh.
        submesh_cells_for_facet = list(dolfin.cells(submesh_facet))
        if len(submesh_cells_for_facet) != 1:
            continue

        # Find how many subdomains are connected to this facet.
        #
        # For this we need the fullmesh facet, because the submesh represents
        # one subdomain.
        fullmesh_facet = find_fullmesh_facet(submesh_facet)
        if subdomains is not None:
            # We must check the fullmesh cells connected to just this one
            # facet, to behave correctly also when three or more subdomains
            # meet at a vertex (and our facet happens to have one of its
            # endpoints at that vertex). Consider the vertical facet in:
            #
            #    S1 | S2
            #    -------
            #      S3
            #
            # In the full mesh, submesh_facet is part of the subdomain boundary
            # between S1/S2, but not part of the subdomain boundaries between
            # S1/S3 or S2/S3. However, the fullmesh cell in S1, touching the
            # vertex, has another facet (in the diagram, a horizontal one) that
            # is part of the boundary S1/S3. Similarly, the fullmesh cell
            # touching the vertex in S2 has another facet that is part of the
            # boundary S2/S3.
            #
            subdomains_for_facet = frozenset({subdomains[cell] for cell in dolfin.cells(fullmesh_facet)})
        else:
            # No subdomains. The value doesn't matter, as long as there is
            # exactly one item; then the boundary gets treated as an external
            # boundary.
            subdomains_for_facet = {1}
        assert len(subdomains_for_facet) in {1, 2}

        # Perform the tagging.
        if len(subdomains_for_facet) == 2:  # Subdomain boundary.
            assert subdomains is not None
            # If the facet is on a boundary between subdomains, and a boundary tag
            # has been specified for this particular subdomain pair, tag the facet
            # in the MeshFunction.
            if subdomains_for_facet in boundary_spec:
                boundary_parts[submesh_facet] = boundary_spec[subdomains_for_facet]
        else:  # len(subdomains_for_facet) == 1
            # External boundary. Delegate to the optional callback.
            if callback:
                tag = callback(submesh_facet, fullmesh_facet)
                if tag is not None:
                    boundary_parts[submesh_facet] = tag

    return boundary_parts
