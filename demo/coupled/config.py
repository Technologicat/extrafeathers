# -*- coding: utf-8; -*-
"""Configuration for the coupled problem demo.

Simulation parameters, file paths, etcetera.
"""

from enum import IntEnum

# --------------------------------------------------------------------------------
# Physical

# Keep in mind the Reynolds number, characterizing the flow:
#
#   Re = ρ u L / μ = u L / ν
#
# where
#
#   ν = μ / ρ

# is the kinematic viscosity, [ν] = m² / s. Similarly, the Péclet number of heat transport is
#
#   Pe = u L / α
#
# where α is the thermal diffusivity:
#
#   α = k / (ρ c)
#
# Note [α] = m² / s,  making it in effect a thermal kinematic viscosity.
#
# For this test, the free stream is  u ≈ 1.5 m/s,  and taking L = 2 r = 2 * 0.05 m = 0.1 m,
# where r is the radius of the cylinder that acts as an obstacle to the flow, we have
#
#   Re = 1 * 1.5 * 0.1 / 0.001 = 150
#
# and for heat transport,
#
#   Pe = 1.5 * 0.1 / (1e-3 / (1 * 1)) = 150
#
#
# Specifically, for the two most common everyday fluids, the material properties are:
#
#  air @ 25°C:
#    ρ = 1.184 kg / m³
#    μ = 18.37e-6 Pa s
#    c = 0.7180e3 J / (kg K)   (@ 26.9°C)
#    k = 26.24e-3 W / (m K)
#  water @ 26.7°C:
#    ρ = 996.8 kg / m³
#    μ = 0.847e-3  Pa s
#    c = 4.179e3 J / (kg K)
#    k = 609.30e-3 W / (m K)  (@ 26.9°C)
#
# Therefore, at u = 1.5 m/s, L = 0.1 m:
#   air:    Re ≈ 9.66794e3 ≈ 10 k    and  Pe ≈ 4.85963e3 ≈ 5 k
#   water:  Re ≈ 0.176529e6 ≈ 0.2 M  and  Pe ≈ 1.025511e6 ≈ 1 M
#
# https://www.engineeringtoolbox.com/air-properties-d_156.html
# https://www.engineeringtoolbox.com/air-absolute-kinematic-viscosity-d_601.html
# https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html
# https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html
# https://www.engineeringtoolbox.com/air-properties-viscosity-conductivity-heat-capacity-d_1509.html
#
# https://www.engineeringtoolbox.com/water-properties-d_1258.html
# https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html
# https://www.engineeringtoolbox.com/water-liquid-gas-thermal-conductivity-temperature-pressure-d_2012.html

rho = 1            # density [kg/m³]
mu = 0.001         # dynamic viscosity [Pa s]
c = 1              # specific heat capacity [J/(kg K)]
k = 1e-3           # heat conductivity [W/(m K)]

# Velocity at the center point of the inflow profile.
inflow_max = 1.5  # [m / s]

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
# Only practically useful for simple testing with low Reynolds number flows (Re ~ 150).
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
