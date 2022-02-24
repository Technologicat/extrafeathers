# -*- coding: utf-8; -*-
"""Miscellaneous utilities."""

__all__ = ["mypause",
           "ScalarOrTensor", "istensor",
           "ufl_constant_property"]

from textwrap import dedent
import typing

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ufl

# Matplotlib (3.3.3) has a habit of popping the figure window to top when it is updated using show() or pause(),
# which effectively prevents using the machine for anything else while a simulation is in progress.
#
# To fix this, the suggestion to use the Qt5Agg backend here:
#   https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
#
# didn't help on my system (Linux Mint 20.1). And it is somewhat nontrivial to use a `FuncAnimation` here.
# So we'll use this custom pause function hack instead, courtesy of StackOverflow user @ImportanceOfBeingErnest:
#   https://stackoverflow.com/a/45734500
#
def mypause(interval: float) -> None:
    """Redraw the current figure without stealing focus.

    Works after `plt.show()` has been called at least once.
    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)


ScalarOrTensor = typing.Union[float,
                              ufl.tensors.ListTensor,
                              ufl.tensors.ComponentTensor,
                              ufl.tensoralgebra.Transposed]
def istensor(x: ScalarOrTensor) -> bool:
    """Return whether `x` is an UFL tensor expression."""
    # TODO: correct way to detect tensor?
    return isinstance(x, (ufl.tensors.ListTensor,
                          ufl.tensors.ComponentTensor,
                          ufl.tensoralgebra.Transposed))


# TODO: A bit dubious whether this is worth the learning cost of an abstraction,
# TODO: but the pattern does repeat a lot in the solvers.
def ufl_constant_property(name, doc):
    """Make an property for wrapping a UFL `Constant`.

    The `Constant` will appear to the outside as if it was a regular `float`
    (scalar) or `np.array` (vector or tensor), as appropriate. This is useful
    in solver interfaces.

    For example::

        ufl_constant_property("ρ", doc="Density [kg / m³]")

    internally executes this::

        def _set_ρ(self, ρ: typing.Union[float, np.array]) -> None:
            self._ρ.assign(ρ)
        def _get_ρ(self) -> typing.Union[float, np.array]:
            if self._ρ.value_size() == 1:
                return float(self._ρ)
            return self._ρ.values()
        ρ = property(fget=_get_ρ, fset=_set_ρ)

    and then returns the property object `ρ`, with its docstring set to `doc`.

    Following the common Python convention for the naming of the underlying
    attribute for a property, the UFL `Constant` object for a given property
    `name` is assumed to be stored as `self._name`. For example, `ρ` is stored
    as `self._ρ`. (In your `__init__` method, `self._ρ = Constant(some_value)`.)
    """
    code = dedent(f"""
    def _set_{name}(self, {name}: typing.Union[float, np.array]) -> None:
        self._{name}.assign({name})
    def _get_{name}(self) -> typing.Union[float, np.array]:
        if self._{name}.value_size() == 1:
            return float(self._{name})
        return self._{name}.values()
    {name} = property(fget=_get_{name}, fset=_set_{name})
    """)
    environment = {"typing": typing, "np": np}
    exec(code, environment)
    # Assign the docstring outside the quasiquoted snippet to
    # avoid issues with possible quote characters inside `doc`.
    environment[name].__doc__ = doc
    return environment[name]
