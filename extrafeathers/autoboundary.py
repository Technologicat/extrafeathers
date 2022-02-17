# -*- coding: utf-8; -*-
"""Automatic extractor for boundaries between subdomains, for use with marked submeshes.

Loosely based on the following, and then improved:
    https://fenicsproject.org/qa/13344/boundary-conditions-on-subdomain/
    https://fenicsproject.org/qa/397/how-define-dirichlet-boundary-conditions-with-meshfunction/

We also provide some utility functions on meshes:

 - Convert a `MeshFunction` on a full mesh to the corresponding
   `MeshFunction` on a `SubMesh` extracted from that mesh.
 - Compute the local mesh size `h` and return it as a `MeshFunction`.
"""

__all__ = ["find_subdomain_boundaries", "specialize_meshfunction",
           "meshsize", "meshfunction_to_expression"]

import typing

import numpy as np

import dolfin


def _make_find_fullmesh_cell(fullmesh: dolfin.Mesh,
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

def _make_find_fullmesh_facet(fullmesh: dolfin.Mesh,
                              submesh: typing.Union[dolfin.SubMesh, dolfin.Mesh],
                              find_fullmesh_cell: typing.Optional[typing.Callable] = None) -> typing.Callable:
    """Make a function that maps a facet of `submesh` to the corresponding facet of `fullmesh`.

    If you have already created a cell mapper using `_make_find_fullmesh_cell` (for the same
    combination of `fullmesh` and `submesh`), you can pass that as the optional callable to
    avoid unnecessarily creating another one.
    """
    if fullmesh is submesh:
        def find_fullmesh_facet(submesh_facet: dolfin.Facet) -> dolfin.Facet:
            return submesh_facet
    else:
        # A facet has zero area measure, so to be geometrically robust, we match a cell.
        find_fullmesh_cell = find_fullmesh_cell or _make_find_fullmesh_cell(fullmesh, submesh)
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


def find_subdomain_boundaries(fullmesh: dolfin.Mesh,
                              submesh: typing.Union[dolfin.SubMesh, dolfin.Mesh],
                              subdomains: typing.Optional[dolfin.MeshFunction],
                              boundary_spec: typing.Dict[typing.FrozenSet[int], int],
                              callback: typing.Optional[typing.Callable] = None) -> dolfin.MeshFunction:
    """Find and tag domain and subdomain boundaries in a mesh.

    Produces a `MeshFunction` on the facets of `submesh`, which can then be used in the solver
    to set boundary conditions on `submesh`. This is useful e.g. for FSI (fluid-structure interaction)
    problems, but applicable to just CFD (computational fluid dynamics), too (by just discarding
    the mesh for the solid subdomain).

    Serial mode only. (As of FEniCS 2019, `SubMesh` is not available in MPI mode anyway.)

    `submesh`: The submesh of `fullmesh` for which to produce the boundary-identifying `MeshFunction`.

    `fullmesh`: The full mesh that `submesh` is part of.

                If no subdomains are used, `fullmesh` can be the same as `submesh`.
                This is useful if one wishes to tag the outer boundary only.

    `subdomains`: A `MeshFunction` on cells of `fullmesh`, specifying the subdomains. (See example below.)
                  If no subdomains are used, can be `None`.

    `boundary_spec`: A dictionary that specifies the boundary tag for each desired pair
                     of subdomains. The format is::

                         {frozenset({subdomain_1_tag, subdomain_2_tag}): boundary_tag,
                          ...}

                     The `frozenset(...)` makes the key hashable. Each tag is a nonnegative integer.

                     Several subdomain pairs are allowed to map to the same boundary_tag;
                     just add a separate entry for each subdomain pair.

                     `boundary_spec` can be the empty dict. Then:
                       - If `subdomains is None`, all boundary facets are treated as if they were
                         on the outer boundary.
                       - If `subdomains` is specified, all subdomain boundaries will be ignored.

    `callback`: If specified, it is called for each facet that is on the outer boundary (i.e. on the
                boundary, but no neighboring subdomain exists). Facilitates tagging all boundaries
                of `submesh` in one go; the solver then only needs one `MeshFunction` to identify
                all boundaries when setting boundary conditions.

                The call signature of `callback` is::

                         f(submesh_facet: Facet, fullmesh_facet: Facet) -> Optional[int]

                The return value should be a boundary tag or `None`. If a boundary tag is
                returned, the facet is tagged (in the resulting `MeshFunction`) by that tag.
                If `None` is returned, the facet will not be tagged.
    """
    facet_dim = submesh.topology().dim() - 1
    boundary_parts = dolfin.MeshFunction('size_t', submesh, facet_dim)  # value_type, mesh, dimension, [default_value]
    boundary_parts.set_all(0)

    find_fullmesh_facet = _make_find_fullmesh_facet(fullmesh, submesh)

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
        # For this we need the fullmesh facet, because the submesh represents one subdomain.
        fullmesh_facet = find_fullmesh_facet(submesh_facet)
        if subdomains is not None:
            # We must check the fullmesh cells connected to just this one facet, to behave
            # correctly also when three or more subdomains meet at a vertex (and our facet
            # happens to have one of its endpoints at that vertex). Consider the vertical
            # facet in:
            #
            #    S1 | S2
            #    -------
            #      S3
            #
            # In the full mesh, submesh_facet is part of the subdomain boundary between
            # S1/S2, but not part of the subdomain boundaries between S1/S3 or S2/S3.
            # However, the fullmesh cell in S1, touching the vertex, has another facet
            # (in the diagram, a horizontal one) that is part of the boundary S1/S3.
            # Similarly, the fullmesh cell touching the vertex in S2 has another facet
            # that is part of the boundary S2/S3.
            #
            subdomains_for_facet = frozenset({subdomains[cell] for cell in dolfin.cells(fullmesh_facet)})
        else:
            # No subdomains. The value doesn't matter, as long as there is exactly one item;
            # then the boundary gets treated as an external boundary.
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


def specialize_meshfunction(f: dolfin.MeshFunction,
                            submesh: dolfin.SubMesh) -> dolfin.MeshFunction:
    """Convert `MeshFunction` `f` on a full mesh to a corresponding `MeshFunction` on `submesh`.

    `submesh` must be a `SubMesh` of `f.mesh()`.

    Supports cell and facet meshfunctions.
    """
    fullmesh = f.mesh()
    dim = fullmesh.topology().dim()
    if f.dim() not in (dim, dim - 1):  # cell and facet functions accepted
        raise NotImplementedError(f"Only cell and facet meshfunctions currently supported; got function of dimension {f.dim()} on mesh of topological dimension {dim}.")

    # TODO: verify that `submesh` is a `SubMesh` of `fullmesh`

    # HACK: map meshfunction classes to the names used by the constructor
    mf_objtype_to_name = {type: name for name, type in dolfin.mesh.meshfunction._meshfunction_types.items()}
    if type(f) not in mf_objtype_to_name:
        # see dolfin/mesh/meshfunction.py
        raise KeyError("MeshFunction type not recognised")

    # Create a new MeshFunction, of the same dimension and type, but on `submesh`.
    g = dolfin.MeshFunction(mf_objtype_to_name[type(f)], submesh, f.dim())  # value_type, mesh, dimension, [default_value]
    g.set_all(0)

    find_fullmesh_cell = _make_find_fullmesh_cell(fullmesh, submesh)
    find_fullmesh_facet = _make_find_fullmesh_facet(fullmesh, submesh, find_fullmesh_cell)

    if f.dim() == dim:  # MeshFunction on cells
        entities = dolfin.cells(submesh)
        find_on_fullmesh = find_fullmesh_cell
    else:  # MeshFunction on facets
        entities = dolfin.facets(submesh)
        find_on_fullmesh = find_fullmesh_facet

    # Copy data from `f`, re-indexing it onto the corresponding submesh entities.
    for submesh_entity in entities:
        fullmesh_entity = find_on_fullmesh(submesh_entity)
        g[submesh_entity] = f[fullmesh_entity]

    return g


def meshsize(mesh: dolfin.Mesh,
             kind: str = "cell") -> dolfin.MeshFunction:
    """Return a `MeshFunction` that gives the local meshsize `h` on each cell or facet of `mesh`.

    kind: "cell" or "facet"
    """
    if kind not in ("cell", "facet"):
        raise ValueError(f"`kind` must be 'cell' or 'facet', got {type(kind)} with value {kind}")

    dim = mesh.topology().dim()
    if kind == "cell":
        entities = dolfin.cells(mesh)
        fdim = dim
    else:  # kind == "facet":
        entities = dolfin.facets(mesh)
        fdim = dim - 1

    f = dolfin.MeshFunction("double", mesh, fdim)
    f.set_all(0.0)

    def vertices_as_array(entity):
        return [vtx.point().array() for vtx in dolfin.vertices(entity)]
    def euclidean_distance(vtxpair):
        assert len(vtxpair) == 2
        dx = vtxpair[0] - vtxpair[1]
        return np.sqrt(np.sum(dx**2))

    # TODO: dolfin::Cell.h() in the C++ API? Is that available in Python?
    # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d2/d12/classdolfin_1_1Cell.html
    for entity in entities:
        edges = dolfin.edges(entity)
        vtxpairs = [vertices_as_array(edge) for edge in edges]
        edge_lengths = [euclidean_distance(vtxpair) for vtxpair in vtxpairs]
        f[entity] = max(edge_lengths)

    return f


def meshfunction_to_expression(f: dolfin.MeshFunction):
    """Convert a scalar double `MeshFunction` to a `CompiledExpression` for use in UFL forms.

    This convenience function mainly exists to document how it's done in FEniCS 2019.

    Based on the tensor-weighted Poisson example:
         https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/tensor-weighted-poisson/demo_tensor-weighted-poisson.py
    """
    # See also:
    #   https://fenicsproject.discourse.group/t/compiledsubdomain-using-c-class/918
    #   https://fenicsproject.org/olddocs/dolfin/latest/cpp/classes.html
    #   https://fenicsproject.org/olddocs/dolfin/latest/cpp/d1/d2e/classdolfin_1_1Expression.html
    cpp_code = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <math.h>
    namespace py = pybind11;

    #include <dolfin/function/Expression.h>
    #include <dolfin/mesh/MeshFunction.h>

    class CellScalarMeshFunctionExpression : public dolfin::Expression
    {
    public:

      virtual void eval(Eigen::Ref<Eigen::VectorXd> values,
                        Eigen::Ref<const Eigen::VectorXd> x,
                        const ufc::cell& cell) const override
      {
        values[0] = (*meshfunction)[cell.index];
      }

      std::shared_ptr<dolfin::MeshFunction<double>> meshfunction;
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
      py::class_<CellScalarMeshFunctionExpression, std::shared_ptr<CellScalarMeshFunctionExpression>, dolfin::Expression>
        (m, "CellScalarMeshFunctionExpression")
        .def(py::init<>())
        .def_readwrite("meshfunction", &CellScalarMeshFunctionExpression::meshfunction);
    }
    """
    return dolfin.CompiledExpression(dolfin.compile_cpp_code(cpp_code).CellScalarMeshFunctionExpression(),
                                     meshfunction=f,
                                     degree=0)
