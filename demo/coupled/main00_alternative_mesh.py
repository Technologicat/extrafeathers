# -*- coding: utf-8; -*-

import pathlib

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import meshfunction
from extrafeathers import meshiowrapper
from extrafeathers import plotmagic

from .config import mesh_filename, Domains


print(pathlib.Path.cwd())

meshiowrapper.import_gmsh(src="demo/meshes/flow_over_two_cylinders.msh",
                          dst="demo/meshes/flow_over_cylinder_full.h5")  # for use by the flow solvers
mesh, domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh("demo/meshes/flow_over_cylinder_full.h5")
fluid_mesh = dolfin.SubMesh(mesh, domain_parts, Domains.FLUID.value)
fluid_domain_parts = meshfunction.specialize(domain_parts, fluid_mesh)
fluid_boundary_parts = meshfunction.specialize(boundary_parts, fluid_mesh)
meshiowrapper.write_hdf5_mesh(mesh_filename,
                              fluid_mesh, fluid_domain_parts, fluid_boundary_parts)

plt.figure(1)
plt.clf()

# mesh itself
plt.subplot(2, 2, 1)
dolfin.plot(mesh)
plt.ylabel("Mesh")

# local mesh size
plt.subplot(2, 2, 2)
theplot = dolfin.plot(meshfunction.meshsize(mesh))
plt.colorbar(theplot)
plt.ylabel("Local mesh size")

# domain parts (subdomains)
plt.subplot(2, 2, 3)
theplot = dolfin.plot(domain_parts)
plt.colorbar(theplot)
plt.ylabel("Phys. surfaces")

# boundary parts
plt.subplot(2, 2, 4)
plotmagic.plot_facet_meshfunction(boundary_parts, invalid_values=[2**64 - 1])
plt.axis("scaled")
plt.legend(loc="best")
plt.ylabel("Phys. boundaries")

plt.suptitle("Fluid")

plt.show()
