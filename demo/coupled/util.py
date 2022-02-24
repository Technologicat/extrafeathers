# -*- coding: utf-8; -*-
"""Miscellaneous utilities."""

__all__ = ["ScalarOrTensor", "istensor",
           "ufl_constant_property"]

from textwrap import dedent
import typing

import numpy as np

import ufl


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
