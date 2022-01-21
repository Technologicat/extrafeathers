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

from extrafeathers import meshutil
from extrafeathers import plotutil

print(pathlib.Path.cwd())

# --------------------------------------------------------------------------------
# Perform conversion
meshutil.import_gmsh(src="demo/flow_over_cylinder.msh",
                     dst="demo/navier_stokes/flow_over_cylinder.h5")

# --------------------------------------------------------------------------------
# Read the result back in
with dolfin.HDF5File(dolfin.MPI.comm_world, "demo/navier_stokes/flow_over_cylinder.h5", "r") as hdf:
    # Meshes can be read back like this:
    mesh = dolfin.Mesh()
    hdf.read(mesh, "/mesh", False)  # target_object, data_path_in_hdf, use_existing_partitioning_if_any

    # For the tags, we must specify which mesh the MeshFunction belongs to, and the function's cell dimension.
    domain_parts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    hdf.read(domain_parts, "/domain_parts")

    boundary_parts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
    hdf.read(boundary_parts, "/boundary_parts")

# --------------------------------------------------------------------------------
# Visualize

# Any facet not belonging to boundary_parts is tagged with a large number:
# size_t(-1) = 2**64 - 1 = 18446744073709551615
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/35
plt.figure(1)
plt.clf()

plt.subplot(3, 1, 1)
dolfin.plot(mesh)
plt.ylabel("Mesh")

plt.subplot(3, 1, 2)
theplot = dolfin.plot(domain_parts)
plt.colorbar(theplot)
plt.ylabel("Phys. surfaces")

plt.subplot(3, 1, 3)
plotutil.plot_facet_meshfunction(boundary_parts, invalid_values=[2**64 - 1])
plt.axis("scaled")
plt.legend(loc="best")
plt.ylabel("Phys. boundaries")

plt.show()
