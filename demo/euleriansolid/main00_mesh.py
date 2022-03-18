# -*- coding: utf-8; -*-

import pathlib

import matplotlib.pyplot as plt

import dolfin

from extrafeathers import meshfunction
from extrafeathers import meshiowrapper
from extrafeathers import plotmagic

print(pathlib.Path.cwd())

meshiowrapper.import_gmsh(src="demo/meshes/box.msh",
                          dst="demo/meshes/box.h5")  # for use by the flow solvers
mesh, domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh("demo/meshes/box.h5")

# Visualize the fluid mesh
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

plt.suptitle("Structure")

plt.show()
