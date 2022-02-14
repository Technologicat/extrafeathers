# -*- coding: utf-8; -*-
"""Advection-diffusion equation: heat transport in a moving material.

ρ c [∂u/∂t + (a·∇)u] - ∇·(k ∇u) = σ : ∇a + ρ h
"""

__all__ = ["AdvectionDiffusion"]

import typing

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, DirichletBC,
                    Function, TrialFunction, TestFunction,
                    Constant, FacetNormal,
                    dot, inner, nabla_grad, div, dx, ds,
                    lhs, rhs, assemble, solve,
                    begin, end)
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

class AdvectionDiffusion:
    """Advection-diffusion equation: heat transport in a moving material.

    `V`: function space for temperature
    `ρ`: density [kg / m³]
    `c`: specific heat capacity [J / (kg K)]
    `k`: heat conductivity [W / (m K)]
         Scalar `k` for thermally isotropic material; rank-2 tensor for anisotropic.
    `dt`: timestep [s]

    `advection`: one of "off", "divergence-free", "general"
        "off":             Zero advection velocity. Standard heat equation.
                           Discards the advection term; runs faster.

        "divergence-free": Treat the advection velocity as divergence-free,
                           discretizing it in a skew-symmetric form that helps
                           the stability of the time integrator.

                           Useful especially when solving heat transport in an
                           incompressible flow.

        "general":         Arbitrary advection velocity.

    The specific heat source `self.h` [W / kg] and the advection velocity `self.a` [m / s]
    are assignable FEM functions. For example, to set a constant heat source everywhere::

        h: Function = interpolate(Constant(1.0), V)
        solver.h.assign(h)
    """
    def __init__(self, V: FunctionSpace,
                 ρ: float, c: float, k: ScalarOrTensor,
                 bc: typing.List[DirichletBC],
                 dt: float,
                 advection: str = "divergence-free"):
        if advection not in ("off", "divergence-free", "general"):
            raise ValueError(f"`advection` must be one of 'off', 'divergence-free', 'general'; got {type(advection)} with value {advection}")

        self.mesh = V.mesh()

        self.V = V
        self.bc = bc

        # Trial and test functions
        self.u = TrialFunction(V)  # no suffix: the UFL symbol for the unknown quantity
        self.v = TestFunction(V)

        # Functions for solution at previous and current time steps
        self.u_n = Function(V)  # suffix _n: the old value (end of previous timestep)
        self.u_ = Function(V)  # suffix _: the latest computed value

        # Specific heat source
        self.h = Function(V)
        self.h.vector()[:] = 0.0  # placeholder value

        # Convection velocity
        #
        # We have this as a general FEM function so that we can feed it with a solution
        # of another subproblem (e.g. the velocity field from the Navier-Stokes solver).
        # TODO: can we create this in Einstein notation? See `ufl.Index`, `ufl.indices`,
        # and https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html
        self.advection = advection

        # dim = self.mesh.topology().dim()
        # self.a = as_vector([Function(V) for j in range(dim)])
        # for aj in self.a:
        #     aj.vector()[:] = 0.0
        # self.σ = as_tensor([[Function(V) for j in range(dim)] for k in range(dim)])
        # print(self.σ)
        # TODO: how to initialize a tensor to zero? How to iterate a tensor over all its components?
        # for σjk in self.σ:
        #     σjk.vector()[:] = 0.0

        V_vec = VectorFunctionSpace(self.mesh, V.ufl_element().family(), V.ufl_element().degree())
        self.a = Function(V_vec)
        self.a.vector()[:] = 0.0

        # Stress
        V_ten = TensorFunctionSpace(self.mesh, V.ufl_element().family(), V.ufl_element().degree())
        self.σ = Function(V_ten)
        self.σ.vector()[:] = 0.0

        # Parameters.
        #
        # We must initialize the underlying variables for properties directly
        # to avoid triggering the compile before all necessary parameters are
        # initialized.
        self._ρ = ρ
        self._c = c
        self._k = k
        self._dt = dt
        self.compile_forms()

    def _set_ρ(self, ρ: float) -> None:
        self._ρ = ρ
        self.compile_forms()
    def _get_ρ(self) -> float:
        return self._ρ
    ρ = property(fget=_get_ρ, fset=_set_ρ, doc="Density [kg / m³]")

    def _set_c(self, c: float) -> None:
        self._c = c
        self.compile_forms()
    def _get_c(self) -> float:
        return self._c
    c = property(fget=_get_c, fset=_set_c, doc="Specific heat capacity [J / (kg K)]")

    def _set_k(self, k: ScalarOrTensor) -> None:
        self._k = k
        self.compile_forms()
    def _get_k(self) -> float:
        return self._k
    k = property(fget=_get_k, fset=_set_k, doc="Heat conductivity [W / (m K)]")

    def _set_dt(self, dt: float) -> None:
        self._dt = dt
        self.compile_forms()
    def _get_dt(self) -> float:
        return self._dt
    dt = property(fget=_get_dt, fset=_set_dt, doc="Timestep [s]")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Temperature
        u = self.u  # new (unknown)
        v = self.v  # test
        u_n = self.u_n  # old (end of previous timestep)

        # Convection velocity
        a = self.a

        # Specific heat source
        h = self.h

        # Stress
        σ = self.σ

        # Wrap constant parameters in a Constant to allow changing the value without triggering a recompile
        ρ = Constant(self.ρ)
        c = Constant(self.c)
        k = Constant(self.k)
        dt = Constant(self.dt)

        # Internal energy balance for a moving material, assuming no phase changes.
        # The unknown `u` is the temperature:
        #
        #   ρ c [∂u/∂t + (a·∇)u] - ∇·(k ∇u) = σ : ∇a + ρ h
        #
        # Derived from the general internal energy balance equation, see Allen et al. (1988).
        #
        # References:
        #     Myron B. Allen, III, Ismael Herrera, and George F. Pinder. 1988.
        #     Numerical Modeling in Science and Engineering. Wiley Interscience.
        U = 0.5 * (u_n + u)  # IMR
        dudt = (u - u_n) / dt
        F = (ρ * c * dudt * v * dx -
             ρ * dot(h, v) * dx)

        if istensor(self._k):
            # TODO: add support for nonzero Neumann BCs
            F += dot(dot(k, nabla_grad(U)), nabla_grad(v)) * dx
            # F += (dot(dot(k, nabla_grad(U)), nabla_grad(v)) * dx -
            #       dot(n, dot(k, nabla_grad(U))) * v * ds)
        else:
            # TODO: add support for nonzero Neumann BCs
            F += k * dot(nabla_grad(U), nabla_grad(v)) * dx
            # F += (k * dot(nabla_grad(U), nabla_grad(v)) * dx -
            #       k * dot(n, nabla_grad(U)) * v * ds)

        if self.advection == "divergence-free":
            # Skew-symmetric form for divergence-free advection velocity (see navier_stokes.py).
            F += (ρ * c * (1 / 2) * (dot(a, nabla_grad(U)) * v - dot(a, nabla_grad(v)) * U) * dx +
                  ρ * c * (1 / 2) * dot(n, a) * U * v * ds)
            F += -inner(σ, nabla_grad(a)) * v * dx  # stress
        elif self.advection == "general":
            # Skew-symmetric, but subtract the contribution from the extra term  (∇·a) u.
            F += (ρ * c * (1 / 2) * (dot(a, nabla_grad(U)) * v - dot(a, nabla_grad(v)) * U) * dx +
                  ρ * c * (1 / 2) * dot(n, a) * U * v * ds -
                  ρ * c * (1 / 2) * div(a) * U * v * dx)
            # # Just use the asymmetric advection term as-is.
            # F += dot(a, nabla_grad(U)) * v * dx
            F += -inner(σ, nabla_grad(a)) * v * dx  # stress

        self.aform = lhs(F)
        self.Lform = rhs(F)

    def step(self) -> None:
        """Take a timestep of length `self.dt`.

        Updates `self.u_`.
        """
        begin("Temperature")
        # TODO: reassemble only the time-varying part of the LHS (advection)
        A = assemble(self.aform)
        b = assemble(self.Lform)
        [bc.apply(A) for bc in self.bc]
        [bc.apply(b) for bc in self.bc]
        solve(A, self.u_.vector(), b, 'bicgstab', 'hypre_amg')
        end()

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This copies `self.u_` to `self.u_n`, making the latest computed solution
        the "old" solution for the next timestep. The old "old" solution is discarded.
        """
        self.u_n.assign(self.u_)
