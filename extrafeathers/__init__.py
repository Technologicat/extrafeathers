# -*- coding: utf-8; -*-
"""Extra batteries for FEniCS for some common tasks.

See individual submodules for more information.

There is also a subpackage `extrafeathers.pdes`, containing some modular
ready-made solvers for some common PDEs, for easily building weakly coupled
multiphysics simulations.

The solvers are mostly intended for use by the `extrafeathers` demos, but
because they can be useful elsewhere, too, they have been included (and are
installed) as a subpackage. However, the subpackage is not automatically
loaded; if you need it, import it explicitly.
"""

__version__ = "0.5.0"

# export the public API
from .autoboundary import *  # noqa: F401, F403
from .meshfunction import *  # noqa: F401, F403
from .meshiowrapper import *  # noqa: F401, F403
from .meshmagic import *  # noqa: F401, F403
from .plotmagic import *  # noqa: F401, F403
