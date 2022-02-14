# -*- coding: utf-8; -*-
"""Simulation parameters, file paths, and similar."""

# physical
rho = 1            # density [kg/m³]
mu = 0.001         # dynamic viscosity [Pa s]
c = 1              # specific heat capacity [J/(kg K)]
k = 1e-3           # heat conductivity [W/(m K)]

# numerical
T = 5.0            # final time [s]
nt = 2500          # number of timesteps
dt = T / nt        # timestep size [s]

# paths

# The solvers expect to be run from the top level of the project as e.g.
#   python -m demo.coupled.main01_flow
# or
#   mpirun python -m demo.coupled.main01_flow
# so the CWD is expected to be the top level, hence the "demo/" at the
# beginning of each path.

# For visualization in ParaView
vis_u_filename = "demo/output/coupled/velocity.xdmf"
vis_p_filename = "demo/output/coupled/pressure.xdmf"
vis_T_filename = "demo/output/coupled/temperature.xdmf"

# For loading into other solvers written using FEniCS. The file extension `.h5` is added automatically.
sol_u_filename = "demo/output/coupled/velocity_series"
sol_p_filename = "demo/output/coupled/pressure_series"
sol_T_filename = "demo/output/coupled/temperature_series"
