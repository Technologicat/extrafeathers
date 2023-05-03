# -*- coding: utf-8; -*-
"""Generate a very simplified uniform mesh for testing, and tag its boundaries."""

import typing

import numpy as np
import matplotlib.pyplot as plt

import dolfin

from extrafeathers import autoboundary
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic, plotmagic

from .config import mesh_filename, Boundaries, L, aspect

def main():
    assert dolfin.MPI.comm_world.size == 1, "Mesh can only be generated in serial mode, please run without mpirun."

    N = 32

    # https://fenicsproject.org/olddocs/dolfin/latest/cpp/de/dac/classdolfin_1_1UnitSquareMesh.html#a0cff260683cd6632fa569ddda6529a0d
    # 'std::string diagonal ("left", "right", "right/left", "left/right", or "crossed")
    #  indicates the direction of the diagonals.'
    # mesh = dolfin.UnitSquareMesh(N, N)
    # mesh = dolfin.UnitSquareMesh(N, N, "crossed")
    # mesh = dolfin.UnitSquareMesh.create(N, N, dolfin.CellType.Type.quadrilateral)
    # mesh = meshmagic.trimesh(N, N)  # rows of equilateral triangles
    # mesh = meshmagic.trimesh(N, N, align="y")  # columns of equilateral triangles
    # from dolfin import ALE, Constant
    # ALE.move(mesh, Constant((-0.5, -0.5)))

    # Aspect ratio (= width / height) of domain.
    # `N` should be integer-divisible by this.
    def vtxpreproc(vtxs):
        # center mesh on origin
        vtxs[:, 0] -= 0.5
        vtxs[:, 1] -= 0.5
        # scale uniformly to desired domain length
        vtxs *= L
        # scale `y` to account for desired aspect ratio
        vtxs[:, 1] /= aspect
        return vtxs
    mesh = meshmagic.trimesh(nx=N, ny=N // aspect, align="y", vtxpreproc=vtxpreproc)
    # mesh = meshmagic.trimesh(nx=N, ny=N // aspect, align="x", vtxpreproc=vtxpreproc)

    domain_parts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

    if mesh.cell_name() == "quadrilateral":
        V = dolfin.FunctionSpace(mesh, "Q", 1)
    else:
        V = dolfin.FunctionSpace(mesh, "P", 1)
    ignored_cells, nodes_dict = meshmagic.all_cells(V)
    ignored_dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])
    print(f"Generated mesh with {nodes_array.shape[0]} vertices, {len(ignored_cells)} cells.")

    # `autoboundary` calls the callback for each facet on the external boundary
    # (i.e. the facet is on a boundary, and no neighboring subdomain exists).
    def autoboundary_callback(submesh_facet: dolfin.Facet, fullmesh_facet: dolfin.Facet) -> typing.Optional[int]:
        p = submesh_facet.midpoint()
        x, y = p.x(), p.y()
        on_vert_boundary = dolfin.near(y, ymin) or dolfin.near(y, ymax)
        on_horz_boundary = dolfin.near(x, xmin) or dolfin.near(x, xmax)
        if dolfin.near(x, xmin) and not on_vert_boundary:
            return Boundaries.LEFT.value
        elif dolfin.near(x, xmax) and not on_vert_boundary:
            return Boundaries.RIGHT.value
        elif dolfin.near(y, ymax) and not on_horz_boundary:
            return Boundaries.TOP.value
        elif dolfin.near(y, ymin) and not on_horz_boundary:
            return Boundaries.BOTTOM.value
        return None  # this facet is not on a boundary we are interested in

    # Tag the boundaries.
    print("Tagging boundaries...")
    boundary_parts: dolfin.MeshFunction = autoboundary.find_subdomain_boundaries(fullmesh=mesh,
                                                                                 submesh=mesh,
                                                                                 subdomains=domain_parts,
                                                                                 boundary_spec={},
                                                                                 callback=autoboundary_callback)

    print("Writing HDF5 file...")
    meshiowrapper.write_hdf5_mesh(mesh_filename, mesh, None, boundary_parts)

    print("Showing mesh.")
    plotmagic.mpiplot_mesh(V)
    plt.axis("equal")
    plotmagic.plot_facet_meshfunction(boundary_parts, names=Boundaries)
    plt.legend(loc="best")
    plt.title("Generated mesh")
    plt.show()
    print("All done.")

if __name__ == "__main__":
    main()
