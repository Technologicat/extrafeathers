# -*- coding: utf-8; -*-
"""Generate a very simplified uniform mesh for testing, and tag its boundaries."""

import typing

import numpy as np
import matplotlib.pyplot as plt

from dolfin import UnitSquareMesh, MeshFunction, Facet, near, MPI

from extrafeathers import autoboundary
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic, plotmagic

from .config import mesh_filename, Boundaries

def main():
    assert MPI.comm_world.size == 1, "Mesh can only be generated in serial mode, please run without mpirun."

    N = 32
    mesh = UnitSquareMesh(N, N)
    from dolfin import ALE, Constant, FunctionSpace
    ALE.move(mesh, Constant((-0.5, -0.5)))

    domain_parts = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

    V = FunctionSpace(mesh, "P", 1)
    ignored_cells, nodes_dict = meshmagic.all_cells(V)
    ignored_dofs, nodes_array = meshmagic.nodes_to_array(nodes_dict)
    xmin = np.min(nodes_array[:, 0])
    xmax = np.max(nodes_array[:, 0])
    ymin = np.min(nodes_array[:, 1])
    ymax = np.max(nodes_array[:, 1])

    # `autoboundary` calls the callback for each facet on the external boundary
    # (i.e. the facet is on a boundary, and no neighboring subdomain exists).
    def autoboundary_callback(submesh_facet: Facet, fullmesh_facet: Facet) -> typing.Optional[int]:
        p = submesh_facet.midpoint()
        x, y = p.x(), p.y()
        on_vert_boundary = near(y, ymin) or near(y, ymax)
        on_horz_boundary = near(x, xmin) or near(x, xmax)
        if near(x, xmin) and not on_vert_boundary:
            return Boundaries.LEFT.value
        elif near(x, xmax) and not on_vert_boundary:
            return Boundaries.RIGHT.value
        elif near(y, ymax) and not on_horz_boundary:
            return Boundaries.TOP.value
        elif near(y, ymin) and not on_horz_boundary:
            return Boundaries.BOTTOM.value
        return None  # this facet is not on a boundary we are interested in

    # Tag the boundaries.
    boundary_parts: MeshFunction = autoboundary.find_subdomain_boundaries(fullmesh=mesh,
                                                                          submesh=mesh,
                                                                          subdomains=domain_parts,
                                                                          boundary_spec={},
                                                                          callback=autoboundary_callback)

    meshiowrapper.write_hdf5_mesh(mesh_filename, mesh, None, boundary_parts)

    from fenics import plot
    plot(mesh)
    plotmagic.plot_facet_meshfunction(boundary_parts, names=Boundaries)
    plt.legend(loc="best")
    plt.title("Generated mesh")
    plt.show()

if __name__ == "__main__":
    main()
