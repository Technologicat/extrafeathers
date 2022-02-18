# -*- coding: utf-8; -*-
"""Configuration for the coupled problem demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical

# Reynolds number?
#
#   Re = ρ u L / μ
#
# Typically for this test, u ≈ 2 m/s,  and taking L = 2 r = 2 * 0.05 m = 0.1 m,
# where r is the radius of the cylinder that acts as an obstacle to the flow,
# we have
#
#   Re = 1 * 2 * 0.1 / 0.001 = 200

rho = 1            # density [kg/m³]
mu = 0.001         # dynamic viscosity [Pa s]
c = 1              # specific heat capacity [J/(kg K)]
k = 1e-3           # heat conductivity [W/(m K)]

# --------------------------------------------------------------------------------
# Numerical

T = 5.0            # final time [s]
nt = 2500          # number of timesteps
dt = T / nt        # timestep size [s]

# --------------------------------------------------------------------------------
# Mesh

# Characteristic length scale for computing the Reynolds number.
# For flow over a cylinder, this is the diameter of the cylinder.
L = 0.1

# These ID numbers must match the numbering in the .msh file (see the .geo file)
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

# Mesh generation
#
# These parameters are only used when generating a uniform mesh using main00_mesh.py.
#
xmin, xmax = 0.0, 2.2
half_height = 0.2
xcyl, ycyl, rcyl = 0.2, 0.2, 0.05
ymin = ycyl - half_height
ymax = ycyl + half_height + 0.01  # asymmetry to excite von Karman vortex street

mesh_resolution = 128

# --------------------------------------------------------------------------------
# File paths

# The solvers expect to be run from the top level of the project as e.g.
#   python -m demo.coupled.main01_flow
# or
#   mpirun python -m demo.coupled.main01_flow
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/meshes/flow_over_cylinder_fluid.h5"  # for input and output

# For visualization in ParaView
vis_u_filename = "demo/output/coupled/velocity.xdmf"
vis_p_filename = "demo/output/coupled/pressure.xdmf"
vis_T_filename = "demo/output/coupled/temperature.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/coupled/velocity_series"
sol_p_filename = "demo/output/coupled/pressure_series"
sol_T_filename = "demo/output/coupled/temperature_series"
