# -*- coding: utf-8; -*-

import pathlib

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import autoboundary
from extrafeathers import meshutil
from extrafeathers import plotutil

print(pathlib.Path.cwd())

meshutil.import_gmsh(src="demo/meshes/flow_over_two_cylinders.msh",
                     dst="demo/meshes/flow_over_cylinder_full.h5")  # for use by the flow solvers
mesh, domain_parts, boundary_parts = meshutil.read_hdf5_mesh("demo/meshes/flow_over_cylinder_full.h5")
fluid_mesh = dolfin.SubMesh(mesh, domain_parts, 5)
fluid_domain_parts = autoboundary.specialize_meshfunction(domain_parts, fluid_mesh)
fluid_boundary_parts = autoboundary.specialize_meshfunction(boundary_parts, fluid_mesh)
meshutil.write_hdf5_mesh("demo/meshes/flow_over_cylinder_fluid.h5",
                         fluid_mesh, fluid_domain_parts, fluid_boundary_parts)

plt.figure(1)
plt.clf()

# mesh itself
plt.subplot(2, 2, 1)
dolfin.plot(mesh)
plt.ylabel("Mesh")

# local mesh size
plt.subplot(2, 2, 2)
theplot = dolfin.plot(autoboundary.meshsize(mesh))
plt.colorbar(theplot)
plt.ylabel("Local mesh size")

# domain parts (subdomains)
plt.subplot(2, 2, 3)
theplot = dolfin.plot(domain_parts)
plt.colorbar(theplot)
plt.ylabel("Phys. surfaces")

# boundary parts
plt.subplot(2, 2, 4)
plotutil.plot_facet_meshfunction(boundary_parts, invalid_values=[2**64 - 1])
plt.axis("scaled")
plt.legend(loc="best")
plt.ylabel("Phys. boundaries")

plt.suptitle("Fluid")

plt.show()
