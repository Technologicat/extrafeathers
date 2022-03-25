# -*- coding: utf-8; -*-
"""Generate a very simplified uniform mesh for testing, and tag its boundaries."""

import typing

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import window

from dolfin import UnitSquareMesh, MeshFunction, Facet, near, MPI

from extrafeathers import autoboundary
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic, plotmagic

from .config import mesh_filename, Boundaries

# TODO: move this to meshmagic
def trimesh(nx, ny, align="x"):
    r"""Make this mesh for the unit square (shown here with nx = 3, ny = 2):

    +------------+
    |\  /\  /\  /| \
    | \/  \/  \/ | |
    +------------+ | ny = 2 rows of triangles
    | /\  /\  /\ | |
    |/  \/  \/  \| /
    +------------+
     \__________/
     nx = 3 bases at bottom

    If `nx = ny`, the triangles are equilateral (except the halves at the ends
    of each row).

    If `align="y"`, flip the roles of x and y when generating the mesh, so that
    instead of rows, the triangles will be arranged in columns.
    """
    hx = 1 / nx
    hy = 1 / ny
    def make_row(j):
        if j % 2 == 0:
            # even row of vertices: nx triangle bases
            xs = np.arange(nx + 1) / nx
        else:
            # odd row of vertices: half-triangle at ends, plus nx - 1 triangle bases.
            xs = np.concatenate(([0.0],
                                 np.arange(hx / 2, 1 - hx / 2 + 1e-8, hx),
                                 [1.0]))
        ys = np.ones_like(xs) * (j * hy)
        if align == "x":
            return list(zip(xs, ys))
        else:
            return list(zip(ys, xs))

    vtxs = make_row(0)
    triangles = []
    kbot = 0  # global DOF at the beginning of the bottom of this triangle row
    ktop = len(vtxs)  # global DOF at the beginning of the top of this triangle row
    for j in range(ny):
        more_vtxs = make_row(j + 1)
        klast = ktop + len(more_vtxs) - 1
        vtxs.extend(more_vtxs)  # upper row of vertices for this row of triangles
        if j % 2 == 0:  # even row of triangles
            # +------------+
            # | /\  /\  /\ |
            # |/  \/  \/  \|
            # +------------+

            #  /\
            # /__\ x nx
            row = []
            for kvert, (kbase1, kbase2) in enumerate(window(2, range(kbot, ktop)),
                                                     start=ktop + 1):
                row.append([kbase1, kbase2, kvert])
            assert len(row) == nx
            triangles.extend(row)

            # ____
            # \  /
            #  \/ x (nx - 1), plus the halves at the ends
            row = []
            row.append([kbot, ktop + 1, ktop])  # left end
            for kvert, (kbase1, kbase2) in enumerate(window(2, range(ktop + 1, klast)),
                                                     start=kbot + 1):
                row.append([kvert, kbase2, kbase1])

            row.append([ktop - 1, klast, klast - 1])  # right end
            assert len(row) == nx + 1
            triangles.extend(row)
        else:  # odd row of triangles
            # +------------+
            # |\  /\  /\  /|
            # | \/  \/  \/ |
            # +------------+

            #  /\
            # /__\ x (nx - 1), plus the halves at the ends
            row = []
            row.append([kbot, kbot + 1, ktop])  # left end
            for kvert, (kbase1, kbase2) in enumerate(window(2, range(kbot + 1, ktop - 1)),
                                                     start=ktop + 1):
                row.append([kbase1, kbase2, kvert])
            row.append([ktop - 2, ktop - 1, klast])  # right end
            assert len(row) == nx + 1
            triangles.extend(row)

            # ____
            # \  /
            #  \/ x nx
            row = []
            for kvert, (kbase1, kbase2) in enumerate(window(2, range(ktop, klast + 1)),
                                                     start=kbot + 1):
                row.append([kvert, kbase2, kbase1])

            assert len(row) == nx
            triangles.extend(row)
        kbot = ktop
        ktop = ktop + len(more_vtxs)

    # # DEBUG
    # vtxs = np.array(vtxs)
    # print(vtxs)
    # print(triangles)
    # import matplotlib.tri as mtri
    # tri = mtri.Triangulation(vtxs[:, 0], vtxs[:, 1], triangles=triangles)
    # plt.triplot(tri)
    # plt.axis("equal")
    # plt.show()

    return meshmagic.make_mesh(cells=triangles, dofs=range(len(vtxs)), vertices=vtxs)


def main():
    assert MPI.comm_world.size == 1, "Mesh can only be generated in serial mode, please run without mpirun."

    N = 16

    # mesh = UnitSquareMesh(N, N)
    mesh = trimesh(N, N)  # rows of equilateral triangles
    # mesh = trimesh(N, N, align="y")  # columns of equilateral triangles

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
