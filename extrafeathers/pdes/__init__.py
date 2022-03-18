# -*- coding: utf-8; -*-
"""Solvers for some PDEs.

Modular, for easily building weakly coupled multiphysics simulations.
"""

# export the public API
from .advection_diffusion import *  # noqa: F401, F403
from .eulerian_solid import *  # noqa: F401, F403
from .navier_stokes import *  # noqa: F401, F403
