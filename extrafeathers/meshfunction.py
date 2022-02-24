# -*- coding: utf-8; -*-
"""Utilities related to `dolfin.MeshFunction`."""

__all__ = ["specialize",
           "meshsize",
           "cell_mf_to_expression"]

import numpy as np

import dolfin

from .common import make_find_fullmesh_cell, make_find_fullmesh_facet


def specialize(f: dolfin.MeshFunction,
               submesh: dolfin.SubMesh) -> dolfin.MeshFunction:
    """Convert `MeshFunction` `f` on a full mesh to a corresponding `MeshFunction` on `submesh`.

    `submesh` must be a `SubMesh` of `f.mesh()`.

    Supports cell and facet meshfunctions.

    Serial mode only.
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

    find_fullmesh_cell = make_find_fullmesh_cell(fullmesh, submesh)
    find_fullmesh_facet = make_find_fullmesh_facet(fullmesh, submesh, find_fullmesh_cell)

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

    if kind == "cell":
        for cell in entities:
            f[cell] = cell.h()
    else:  # facets have no `.h`
        def vertices_as_array(entity):
            return [vtx.point().array() for vtx in dolfin.vertices(entity)]
        def euclidean_distance(vtxpair):
            assert len(vtxpair) == 2
            dx = vtxpair[0] - vtxpair[1]
            return np.sqrt(np.sum(dx**2))

        for entity in entities:
            edges = dolfin.edges(entity)
            vtxpairs = [vertices_as_array(edge) for edge in edges]
            edge_lengths = [euclidean_distance(vtxpair) for vtxpair in vtxpairs]
            f[entity] = max(edge_lengths)

    return f


# Based on the tensor-weighted Poisson example. The API has changed once or twice; see the latest:
#    https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/tensor-weighted-poisson/
# See also:
#   https://fenicsproject.org/olddocs/dolfin/latest/cpp/d1/d2e/classdolfin_1_1Expression.html
#   https://fenicsproject.discourse.group/t/compiledsubdomain-using-c-class/918
#   https://fenicsproject.org/olddocs/dolfin/latest/cpp/classes.html
cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <math.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class CellMeshFunctionExpression : public dolfin::Expression
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
  py::class_<CellMeshFunctionExpression, std::shared_ptr<CellMeshFunctionExpression>, dolfin::Expression>
    (m, "CellMeshFunctionExpression")
    .def(py::init<>())
    .def_readwrite("meshfunction", &CellMeshFunctionExpression::meshfunction);
}
"""
_compiled_cpp_code = dolfin.compile_cpp_code(cpp_code)

def cell_mf_to_expression(f: dolfin.MeshFunction):
    """Convert a scalar double `MeshFunction` on cells to a `CompiledExpression` for use in UFL forms.

    This convenience function mainly exists to document how it's done in FEniCS 2019.
    """
    return dolfin.CompiledExpression(_compiled_cpp_code.CellMeshFunctionExpression(),
                                     meshfunction=f,
                                     degree=0)
