# -*- coding: utf-8; -*-
"""Main program for the coupled problem demo.

Zeroth pass main program: generate a mesh, with uniform cell size,
using FEniCS's Mshr API.

Running `python -m demo.import_gmsh` is an alternative to this;
it will import a pre-prepared graded mesh created with Gmsh.
Note these alternatives will overwrite each other's output.

This module also contains some configuration important for the solvers;
especially, the boundary tag IDs, and ymin/ymax for setting up the inflow
velocity profile.
"""

from enum import IntEnum
import typing

import matplotlib.pyplot as plt

from dolfin import Point, MeshFunction, SubMesh, Facet, near, MPI
from mshr import Rectangle, Circle, generate_mesh

from extrafeathers import autoboundary
from extrafeathers import meshutil
from extrafeathers import plotutil


mesh_filename = "demo/meshes/flow_over_cylinder_fluid.h5"  # for input and output

mesh_resolution = 128  # only used during mesh generation

# These numbers must match the numbering in the .msh file (see the .geo file)
# so that the Gmsh-imported mesh works as expected, too.
class Boundaries(IntEnum):
    # Autoboundary always tags internal facets with the value 0.
    # Leave it out from the definitions to make the boundary plotter ignore any facet tagged with that value.
    # NOT_ON_BOUNDARY = 0
    INFLOW = 1
    WALLS = 2
    OUTFLOW = 3
    OBSTACLE = 4
class Domains(IntEnum):
    FLUID = 5
    STRUCTURE = 6

# Geometry parameters
# These must also match the .msh file, because the ymin/ymax values are used for setting up
# the inflow profile in the boundary conditions.
xmin, xmax = 0.0, 2.2
half_height = 0.2
xcyl, ycyl, rcyl = 0.2, 0.2, 0.05
ymin = ycyl - half_height
ymax = ycyl + half_height + 0.01  # asymmetry to excite von Karman vortex street

# # The original single-file example used to define boundaries like this
# # (implicit CompiledSubdomain when fed to DirichletBC):
# inflow = f'near(x[0], {xmin})'
# outflow = f'near(x[0], {xmax})'
# walls = f'near(x[1], {ymin}) || near(x[1], {ymax})'
# cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'
#
# `extrafeathers` provides another way to define boundaries: the `autoboundary` module.
#   - Boundary facets between subdomains are extracted automatically.
#   - External boundary facets can be tagged using a callback.
# This produces one MeshFunction (on submesh facets) that has tags for all boundaries.

def main():
    """Generate the mesh."""
    # Mesh generation
    #
    # As of FEniCS 2019, marked subdomains are not supported in MPI mode,
    # so any mesh generation using them must be done in serial mode.
    # https://fenicsproject.discourse.group/t/mpi-and-subdomains/339/2
    assert MPI.comm_world.size == 1, "Mesh can only be generated in serial mode, please run without mpirun."

    # Create geometry
    #
    # Note we do not cut out the cylinder; we just mark it. This can be used
    # to generate a subdomain boundary, which can then be tagged with the
    # `autoboundary` utility of `extrafeathers`.
    #
    # We at first mark all cells as belonging to the first subdomain,
    # and then mark the other subdomains (we have only one here).
    domain = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    cylinder = Circle(Point(xcyl, ycyl), rcyl)
    domain.set_subdomain(Domains.FLUID.value, domain)
    domain.set_subdomain(Domains.STRUCTURE.value, cylinder)

    mesh = generate_mesh(domain, mesh_resolution)

    # Create mesh function on cells identifying the marked subdomains.
    # This allows us to extract different subdomains as separate submeshes.
    domain_parts = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
    fluid_mesh = SubMesh(mesh, domain_parts, Domains.FLUID.value)
    structure_mesh = SubMesh(mesh, domain_parts, Domains.STRUCTURE.value)

    # Using the submeshes, we can use `autoboundary` to extract and tag the subdomain boundaries.

    # Specify pairs of subdomains that indicate a physically meaningful subdomain boundary.
    autoboundary_spec = {frozenset({Domains.FLUID.value, Domains.STRUCTURE.value}): Boundaries.OBSTACLE.value}

    # `autoboundary` calls the callback for each facet on the external boundary
    # (i.e. the facet is on a boundary, and no neighboring subdomain exists).
    #
    # For a box domain we could do this, too:
    # xmin, ymin, zmin = domain.first_corner().array()
    # xmax, ymax, zmax = domain.second_corne().array()  # "corne" as of FEniCS 2019.
    def autoboundary_callback(submesh_facet: Facet, fullmesh_facet: Facet) -> typing.Optional[int]:
        p = submesh_facet.midpoint()
        x, y = p.x(), p.y()
        if near(x, xmin):
            return Boundaries.INFLOW.value
        elif near(x, xmax):
            return Boundaries.OUTFLOW.value
        elif near(y, ymin) or near(y, ymax):
            return Boundaries.WALLS.value
        return None  # this facet is not on a boundary we are interested in

    # Tag the boundaries.
    fluid_boundary_parts: MeshFunction = autoboundary.find_subdomain_boundaries(fullmesh=mesh,
                                                                                submesh=fluid_mesh,
                                                                                subdomains=domain_parts,
                                                                                boundary_spec=autoboundary_spec,
                                                                                callback=autoboundary_callback)

    # Save meshes, subdomains and boundary data as HDF5
    #
    # Note, however, that our `domain_parts` are specified w.r.t. `mesh`, not `fluid_mesh`,
    # so they are not applicable here. If we wanted, we could `autoboundary.specialize_meshfunction` it
    # onto `fluid_mesh`. But we don't need the subdomain data in this solver, so we can just leave it out.
    meshutil.write_hdf5_mesh(mesh_filename, fluid_mesh, None, fluid_boundary_parts)

    print("Mesh generated, visualizing.")
    print("Please run 01_flow.py and then 02_heat.py to solve the problem.")
    from fenics import plot
    plot(fluid_mesh)
    plot(structure_mesh, color="tan")  # note: not saved to file
    plotutil.plot_facet_meshfunction(fluid_boundary_parts, names=Boundaries)
    plt.legend(loc="best")
    plt.title("Generated mesh")
    plt.show()

if __name__ == "__main__":
    main()
