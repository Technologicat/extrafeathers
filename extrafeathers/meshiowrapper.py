# -*- coding: utf-8; -*-
"""Gmsh import for FEniCS (with subdomain and boundary data), built on top of meshio.

Stores the result as a single HDF5 file.

We also provide convenience functions for I/O of meshes and subdomain/boundary data
as HDF5.

Based on many posts in these discussion threads:
  https://fenicsproject.discourse.group/t/converting-simple-2d-mesh-from-gmsh-to-dolfin/583/4
  https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/9
"""

__all__ = ["read_hdf5_mesh", "write_hdf5_mesh", "import_gmsh"]

import pathlib
import tempfile
import typing

import numpy as np

import meshio

import dolfin

mesh_dataset_name = "mesh"
cell_dataset_name = "domain_parts"
boundary_dataset_name = "boundary_parts"

def _absolute_path(filename_or_path: typing.Union[pathlib.Path, str]) -> pathlib.Path:
    return pathlib.Path(filename_or_path).expanduser().resolve()


def read_hdf5_mesh(filename: str) -> typing.Tuple[dolfin.Mesh,
                                                  dolfin.MeshFunction,
                                                  dolfin.MeshFunction]:
    """Read an HDF5 mesh file created using `write_hdf5_mesh` or `import_gmsh`.

    The return value is `(mesh, domain_parts, boundary_parts)`.

    Raises `ValueError` if the file does not have a "/mesh" dataset.

    The "/domain_parts" and "/boundary_parts" datasets are optional.
    If the file does not have them, the corresponding component of
    the return value will be `None`.
    """
    filename = str(_absolute_path(filename))

    with dolfin.HDF5File(dolfin.MPI.comm_world, filename, "r") as hdf:
        if not hdf.has_dataset(mesh_dataset_name):
            raise ValueError(f"{mesh_dataset_name} dataset not found in mesh file {filename}")

        mesh = dolfin.Mesh()
        hdf.read(mesh, mesh_dataset_name, False)  # target_object, data_path_in_hdf, use_existing_partitioning_if_any

        if hdf.has_dataset(cell_dataset_name):
            # For the tags, we must specify which mesh the MeshFunction belongs to,
            # and the function's cell dimension.
            domain_parts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)  # type, mesh, dim, [default_value]
            hdf.read(domain_parts, cell_dataset_name)
        else:
            domain_parts = None

        if hdf.has_dataset(boundary_dataset_name):
            boundary_parts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
            hdf.read(boundary_parts, boundary_dataset_name)
        else:
            boundary_parts = None

    return mesh, domain_parts, boundary_parts


def write_hdf5_mesh(filename: str,
                    mesh: dolfin.Mesh,
                    domain_parts: typing.Optional[dolfin.MeshFunction] = None,
                    boundary_parts: typing.Optional[dolfin.MeshFunction] = None) -> None:
    """Write a mesh and its associated subdomain/boundary mesh functions into an HDF5 file.

    The optional mesh functions should be a `MeshFunction` of type `size_t`.
    `domain_parts` should be defined on the cells of `mesh`;
    `boundary_parts` should be defined on the facets of `mesh`.
    These are meant for physical tagging of subdomains and boundaries.

    The output is a single HDF5 file with three datasets:
      - "/mesh" contains the mesh itself.
      - "/domain_parts" (if specified) contains the physical cells
        (i.e. surfaces in 2D, volumes in 3D).
      - "/boundary_parts" (if specified) contains the physical facets
        (i.e. lines in 2D, surfaces in 3D).
    """
    filename = str(_absolute_path(filename))

    with dolfin.HDF5File(mesh.mpi_comm(), filename, "w") as hdf:
        hdf.write(mesh, mesh_dataset_name)
        if domain_parts:
            hdf.write(domain_parts, cell_dataset_name)  # MeshFunction on cells of `mesh`
        if boundary_parts:
            hdf.write(boundary_parts, boundary_dataset_name)  # MeshFunction on facets of `mesh`


def import_gmsh(src: typing.Union[pathlib.Path, str],
                dst: typing.Union[pathlib.Path, str]) -> None:
    """Import a Gmsh mesh into FEniCS.

    Physical cells (volumes in 3D, surfaces in 2D) and facets (surfaces in 3D, lines in 2D)
    are also imported.

    Simplicial meshes (triangles/tetrahedrons) in 2D or 3D only.

    `src`: The `.msh` file to import.
    `dst`: The `.h5` file to write.

    If the same cell or facet has multiple physical tags, the latest one wins.

    On the output format, see `write_hdf5_mesh`.
    The file can be read back in using `read_hdf5_mesh`.
    """
    if dolfin.MPI.comm_world.size > 1:
        raise NotImplementedError("`import_gmsh` does not support running in parallel. Perform the mesh file conversion serially, then run your solver in parallel.")

    src = _absolute_path(src)
    dst = _absolute_path(dst)

    output_directory = dst.parent
    output_directory.mkdir(exist_ok=True)

    # print(meshio._helpers.extension_to_filetype)  # formats supported by current installation of meshio

    # Read input mesh (generated by Gmsh)
    msh = meshio.read(src)
    cells_dict = msh.cells_dict  # property; compute it just once

    # Detect the topological dimension of the mesh.
    #
    # HACK: topological_dimension is declared public (in `__all__` of `__init__.py`),
    # but meshio 5.2.2 forgets to actually import it to the top-level namespace.
    cell_topological_dimensions = [meshio._mesh.topological_dimension[kind] for kind in cells_dict]
    dim = max(cell_topological_dimensions)

    if dim not in (2, 3):
        raise NotImplementedError("`import_gmsh` currently only supports meshes with topological dimension 2 or 3.")

    if dim == 2:
        # Force geometric dimension of mesh to 2D by deleting z coordinate
        msh.points = msh.points[:, :2]

    # TODO: add support for quadrilateral/hexahedral meshes
    cell_kind = "triangle" if dim == 2 else "tetra"
    facet_kind = "line" if dim == 2 else "triangle"

    # TODO: A bit clunky; can we simplify this to one step without temporary files?

    # Convert geometry to XDMF
    with tempfile.NamedTemporaryFile() as temp_mesh_out:
        # meshio expects a pathlike, not a buffer directly.
        #
        # "Whether the name can be used to open the file a second time, while the
        # named temporary file is still open, varies across platforms (it can be so
        # used on Unix; it cannot on Windows NT or later)."
        #   --https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
        #
        # As of meshio 5.2.2, recent versions stack cell blocks automatically,
        # so we can just read from `msh.cells_dict`.
        meshio.write(temp_mesh_out.name,
                     meshio.Mesh(points=msh.points,
                                 cells={cell_kind: cells_dict[cell_kind]}),
                     file_format="xdmf")

        # Read the temporary XDMF into FEniCS.
        #
        # XDMFFile is a wrapper for C++ code so it's safe to assume it
        # doesn't support Python filelikes. So pass in the filename here too...
        mesh = dolfin.Mesh()
        with dolfin.XDMFFile(temp_mesh_out.name) as temp_mesh_in:
            temp_mesh_in.read(mesh)
        # print(f"Read mesh with topological dimension {mesh.topology().dim()} and geometric dimension {mesh.geometric_dimension()}.")  # DEBUG

    # Convert Gmsh physical facets (lines in 2D, surfaces in 3D) to XDMF
    #
    # MeshValueCollection represents raw imported data, which can be then
    # loaded into a MeshFunction.
    # https://fenicsproject.discourse.group/t/difference-between-meshvaluecollection-and-meshfunction/5219
    #
    # Note any facet not present in the MeshValueCollection will be tagged
    # with a large number:
    # size_t(-1) = 2**64 - 1 = 18446744073709551615
    #
    # https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/35
    with tempfile.NamedTemporaryFile() as boundary_parts_out:
        meshio.write(boundary_parts_out.name,
                     meshio.Mesh(points=msh.points,
                                 cells={facet_kind: cells_dict[facet_kind]},
                                 cell_data={boundary_dataset_name: [_physical_groups(msh, facet_kind)]}),
                     file_format="xdmf")

        mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
        with dolfin.XDMFFile(boundary_parts_out.name) as boundary_parts_in:
            boundary_parts_in.read(mvc, boundary_dataset_name)
        boundary_parts = dolfin.MeshFunction("size_t", mesh, mvc)

    # Convert Gmsh physical cells (surfaces in 2D, volumes in 3D) to XDMF
    with tempfile.NamedTemporaryFile() as domain_parts_out:
        meshio.write(domain_parts_out.name,
                     meshio.Mesh(points=msh.points,
                                 cells={cell_kind: cells_dict[cell_kind]},
                                 cell_data={cell_dataset_name: [_physical_groups(msh, cell_kind)]}),
                     file_format="xdmf")

        mvc = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
        with dolfin.XDMFFile(domain_parts_out.name) as domain_parts_in:
            domain_parts_in.read(mvc, cell_dataset_name)
        domain_parts = dolfin.MeshFunction("size_t", mesh, mvc)

    # Write the single HDF5 output file
    write_hdf5_mesh(str(dst), mesh, domain_parts, boundary_parts)


# Based on a part of `create_entity_mesh` by @Dokken, at
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/158
def _physical_groups(mesh: meshio.Mesh, kind: str):
    """Extract physical group definitions for use as `cell_data`.

    `kind`: "line", "triangle", or "tetra".
    """
    try:
        # If the mesh was created with the gmsh API, it is simple to extract
        # the entity data.
        cell_data = mesh.cell_data_dict["gmsh:physical"][kind]
    except KeyError:
        # If the mesh was created with pygmsh, we need to parse through cell sets and
        # sort the data.
        cell_entities = []
        cell_data = []
        for marker, entity_set in mesh.cell_sets_dict.items():
            for thekind, entities in entity_set.items():
                if thekind == kind:
                    cell_entities.append(entities)
                    cell_data.append(np.full(len(entities), int(marker)))
        cell_entities = np.hstack(cell_entities)
        sorted_cell_entities = np.argsort(cell_entities)
        cell_data = np.hstack(cell_data)[sorted_cell_entities]
    return cell_data
