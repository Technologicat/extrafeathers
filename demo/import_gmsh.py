# -*- coding: utf-8; -*-
"""Gmsh mesh import demo.

This creates the HDF5 mesh file for the Navier-Stokes demo.

This script should be run from the top level of the project,
*NOT* while inside the `demo/` subdirectory. The incantation is::

    python -m demo.import_gmsh
"""

import pathlib

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import meshfunction
from extrafeathers import meshutil
from extrafeathers import plotutil

print(pathlib.Path.cwd())

# Import the mesh (full domain, containing both fluid and structure)
meshutil.import_gmsh(src="demo/meshes/flow_over_cylinder.msh",
                     dst="demo/meshes/flow_over_cylinder_full.h5")

# Read the result back in
mesh, domain_parts, boundary_parts = meshutil.read_hdf5_mesh("demo/meshes/flow_over_cylinder_full.h5")

# Separate the fluid and structure meshes (in the `navier_stokes` demo, we only need the fluid mesh)
# The tag numbers must match those that were used in the input .msh file (see the .geo file it is generated from).
fluid_mesh = dolfin.SubMesh(mesh, domain_parts, 5)
fluid_domain_parts = meshfunction.specialize(domain_parts, fluid_mesh)
fluid_boundary_parts = meshfunction.specialize(boundary_parts, fluid_mesh)
meshutil.write_hdf5_mesh("demo/meshes/flow_over_cylinder_fluid.h5",
                         fluid_mesh, fluid_domain_parts, fluid_boundary_parts)

structure_mesh = dolfin.SubMesh(mesh, domain_parts, 6)
structure_domain_parts = meshfunction.specialize(domain_parts, structure_mesh)
structure_boundary_parts = meshfunction.specialize(boundary_parts, structure_mesh)
meshutil.write_hdf5_mesh("demo/meshes/flow_over_cylinder_structure.h5",
                         structure_mesh, structure_domain_parts, structure_boundary_parts)

# --------------------------------------------------------------------------------
# Visualize

# Any facet not belonging to boundary_parts is tagged with a large number:
# size_t(-1) = 2**64 - 1 = 18446744073709551615
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/35

for figno, (title, msh, dparts, bparts) in enumerate((("All", mesh,
                                                       domain_parts, boundary_parts),
                                                      ("Fluid", fluid_mesh,
                                                       fluid_domain_parts, fluid_boundary_parts),
                                                      ("Structure", structure_mesh,
                                                       structure_domain_parts, structure_boundary_parts)),
                                                     start=1):
    plt.figure(figno)
    plt.clf()

    # mesh itself
    plt.subplot(2, 2, 1)
    dolfin.plot(msh)
    plt.ylabel("Mesh")

    # local mesh size
    plt.subplot(2, 2, 2)
    theplot = dolfin.plot(meshfunction.meshsize(msh))
    plt.colorbar(theplot)
    plt.ylabel("Local mesh size")

    # domain parts (subdomains)
    plt.subplot(2, 2, 3)
    theplot = dolfin.plot(dparts)
    plt.colorbar(theplot)
    plt.ylabel("Phys. surfaces")

    # boundary parts
    plt.subplot(2, 2, 4)
    plotutil.plot_facet_meshfunction(bparts, invalid_values=[2**64 - 1])
    plt.axis("scaled")
    plt.legend(loc="best")
    plt.ylabel("Phys. boundaries")

    plt.suptitle(title)

plt.show()
