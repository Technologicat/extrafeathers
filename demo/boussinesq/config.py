# -*- coding: utf-8; -*-
"""Configuration for the Boussinesq flow (natural convection) demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical

rho = 1            # density [kg/m³]
mu = 0.001         # dynamic viscosity [Pa s]
c = 1              # specific heat capacity [J/(kg K)]
k = 1e-3           # heat conductivity [W/(m K)]
alpha = 1e-3       # coefficient of thermal expansion [1/K]
T0 = 0             # reference temperature at which thermal expansion is zero [K]
g = 9.81           # acceleration of gravity [m/s²]

# --------------------------------------------------------------------------------
# Numerical

T = 5.0            # final time [s]
nt = 500           # number of timesteps
dt = T / nt        # timestep size [s]

# --------------------------------------------------------------------------------
# Mesh

# Characteristic length scale for computing the Reynolds number.
# In the example, this is the size of the heated object.
L = 0.1

# These ID numbers must match the numbering in the .msh file (see the .geo file)
# so that the Gmsh-imported mesh works as expected, too.
class Boundaries(IntEnum):
    # Autoboundary always tags internal facets with the value 0.
    # Leave it out from the definitions to make the boundary plotter ignore any facet tagged with that value.
    # NOT_ON_BOUNDARY = 0
    WALLS = 1
    OBSTACLE = 2
class Domains(IntEnum):
    FLUID = 3
    STRUCTURE = 4  # not used in simulation

# --------------------------------------------------------------------------------
# File paths

# The solvers expect to be run from the top level of the project as e.g.
#   python -m demo.boussinesq.main01_solve
# or
#   mpirun python -m demo.boussinesq.main01_solve
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

mesh_filename = "demo/meshes/cavity_with_obstacle_fluid.h5"  # for input and output

# For visualization in ParaView
vis_u_filename = "demo/output/boussinesq/velocity.xdmf"
vis_p_filename = "demo/output/boussinesq/pressure.xdmf"
vis_T_filename = "demo/output/boussinesq/temperature.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/boussinesq/velocity_series"
sol_p_filename = "demo/output/boussinesq/pressure_series"
sol_T_filename = "demo/output/boussinesq/temperature_series"
