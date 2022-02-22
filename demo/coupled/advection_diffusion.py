# -*- coding: utf-8; -*-
"""Advection-diffusion equation: heat transport in a moving material.

This is based on the internal energy (enthalpy) balance of a continuum:

  ρ c [∂u/∂t + (a·∇)u] - ∇·(k ∇u) = σ : ∇a + ρ h

where

  - `u` is the absolute temperature [K],
  - `a` is the advection velocity [m / s], and
  - `σ` is the stress tensor [Pa].

This form of the equation is based on Fourier's law, and the modeling
assumption that the specific internal energy `e` [J / kg] behaves as

  e = c u

i.e. there are no phase changes.

Strictly, we also assume that `c` is a constant independent of the
temperature `u`. If  c = c(u),  we must add the term

  ρ ∂c/∂u u [∂u/∂t + (a·∇)u]

to the LHS, and the equation becomes nonlinear; this is currently
not supported by this solver.

See:
    Myron B. Allen, III, Ismael Herrera, and George F. Pinder. 1988.
    Numerical Modeling in Science and Engineering. Wiley Interscience.
"""

__all__ = ["AdvectionDiffusion"]

import typing

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, DirichletBC,
                    Function, TrialFunction, TestFunction, Expression,
                    Constant, FacetNormal,
                    dot, inner, nabla_grad, div, dx, ds,
                    lhs, rhs, assemble, solve,
                    begin, end)
import ufl

from extrafeathers import autoboundary


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
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 gives the implicit midpoint rule; 0 is forward Euler,
         and 1 is backward Euler.

    `advection`: one of "off", "divergence-free", "general"
        "off":             Zero advection velocity. Standard heat equation.
                           Discards the advection term; runs faster.

        "divergence-free": Treat the advection velocity as divergence-free,
                           discretizing it in a skew-symmetric form that helps
                           the stability of the time integrator.

                           Useful especially when solving heat transport in an
                           incompressible flow.

        "general":         Arbitrary advection velocity.

    `velocity_degree`: The degree of the finite element space of the advection
                       velocity field. This must match the data you are loading in.

                       The default `None` means "use the same degree as `V`".

                       Used only when `advection != "off"`.

    `use_stress`: whether to include the stress term  σ : ∇a,  which represents the
                  contribution of stress to internal energy.

                  Due to the factor ∇a, `use_stress` can only be enabled when
                  `advection` is also enabled (i.e. something other than "off").

                  Setting `use_stress=True` is only useful if you intend to send in
                  a stress tensor (possibly at each timestep).

                  When `use_stress=False`, the stress term is discarded; runs faster.

    `stress_degree`: The degree of the finite element space of the stress field.
                     This must match the data you are loading in.

                     The default `None` means "use the same degree as `V`".

                     Used only when `use_stress=True`.

    The specific heat source `self.h` [W / kg], the advection velocity `self.a` [m / s],
    and the stress tensor `self.σ` [Pa] are assignable FEM functions.

    For example, to set a constant heat source everywhere::

        h: Function = interpolate(Constant(1.0), V)
        solver.h.assign(h)

    Anything compatible with a `Function` can be assigned, including FEM fields produced
    by another solver or loaded from a file (that was produced by a solver on the same mesh!).
    Technically, `h` lives on a `FunctionSpace`, `a` on a `VectorFunctionSpace`, and `σ` on
    a `TensorFunctionSpace` compatible with `V`.
    """
    def __init__(self, V: FunctionSpace,
                 ρ: float, c: float, k: ScalarOrTensor,
                 bc: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5, *,
                 advection: str = "divergence-free",
                 velocity_degree: int = None,
                 use_stress: bool = False,
                 stress_degree: int = None):
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

        # Local mesh size (for stabilization terms)
        self.he = autoboundary.cell_meshfunction_to_expression(autoboundary.meshsize(self.mesh))

        # Specific heat source
        self.h = Function(V)
        self.h.vector()[:] = 0.0  # placeholder value

        # We have the convection velocity and stress as general FEM functions
        # so that we can feed them with a solution of another subproblem.

        # Convection velocity
        self.advection = advection
        a_degree = velocity_degree if velocity_degree is not None else V.ufl_element().degree()
        V_vec = VectorFunctionSpace(self.mesh, V.ufl_element().family(), a_degree)
        self.a = Function(V_vec)
        self.a.vector()[:] = 0.0

        # Stress
        self.use_stress = use_stress and self.advection != "off"
        σ_degree = stress_degree if stress_degree is not None else V.ufl_element().degree()
        V_ten = TensorFunctionSpace(self.mesh, V.ufl_element().family(), σ_degree)
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
        self._θ = θ
        self.compile_forms()

    # TODO: should have the `Constant` instance at this level, to make it settable without recompiling forms.
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

    def _set_θ(self, θ: float) -> None:
        self._θ = θ
        self.compile_forms()
    def _get_θ(self) -> float:
        return self._θ
    θ = property(fget=_get_θ, fset=_set_θ, doc="Theta-parameter for time integrator")

    def peclet(self, uinf, L):
        """Return the Péclet number of the heat transport.

        `uinf`: free-stream speed [m / s]
        `L`: length scale [m]
        """
        α = self.k / (self.ρ * self.c)
        return uinf * L / α

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

        # Local mesh size (for stabilization terms)
        he = self.he

        # Wrap constant parameters in a Constant to allow changing the value without triggering a recompile
        ρ = Constant(self.ρ)
        c = Constant(self.c)
        k = Constant(self.k)
        dt = Constant(self.dt)
        θ = Constant(self.θ)

        # Internal energy balance for a moving material, assuming no phase changes.
        # The unknown `u` is the temperature:
        #
        #   ρ c [∂u/∂t + (a·∇)u] - ∇·(k·∇u) = σ : ∇a + ρ h
        #
        # Derived from the general internal energy balance equation, see Allen et al. (1988).
        #
        # References:
        #     Myron B. Allen, III, Ismael Herrera, and George F. Pinder. 1988.
        #     Numerical Modeling in Science and Engineering. Wiley Interscience.
        U = (1 - θ) * u_n + θ * u
        dudt = (u - u_n) / dt

        F = (ρ * c * dudt * v * dx -
             ρ * dot(h, v) * dx)

        if istensor(self._k):
            F += dot(dot(k, nabla_grad(U)), nabla_grad(v)) * dx
            # TODO: add support for nonzero Neumann BCs
            # The full weak form of -∇·(k·∇u) is:
            # F += (dot(dot(k, nabla_grad(U)), nabla_grad(v)) * dx -
            #       dot(n, dot(k, nabla_grad(U))) * v * ds)
        else:
            F += k * dot(nabla_grad(U), nabla_grad(v)) * dx
            # TODO: add support for nonzero Neumann BCs
            # The full weak form of -∇·(k·∇u) is:
            # F += (k * dot(nabla_grad(U), nabla_grad(v)) * dx -
            #       k * dot(n, nabla_grad(U)) * v * ds)

        if self.advection == "divergence-free":
            # Skew-symmetric form for divergence-free advection velocity (see navier_stokes.py).
            F += (ρ * c * (1 / 2) * (dot(a, nabla_grad(U)) * v - dot(a, nabla_grad(v)) * U) * dx +
                  ρ * c * (1 / 2) * dot(n, a) * U * v * ds)
        elif self.advection == "general":
            # Skew-symmetric advection, as above; but subtract the contribution from the extra term
            # (∇·a) u, thus producing an extra symmetric term that accounts for the divergence of `a`.
            F += (ρ * c * (1 / 2) * (dot(a, nabla_grad(U)) * v - dot(a, nabla_grad(v)) * U) * dx +
                  ρ * c * (1 / 2) * dot(n, a) * U * v * ds -
                  ρ * c * (1 / 2) * div(a) * U * v * dx)
            # # Just use the asymmetric advection term as-is.
            # # TODO: which form is better? This has fewer operations, but which gives better stability?
            # F += dot(a, nabla_grad(U)) * v * dx

        if self.use_stress and self.advection != "off":
            F += -inner(σ, nabla_grad(a)) * v * dx

        # SUPG: streamline upwinding Petrov-Galerkin.
        #
        def mag(vec):
            return dot(vec, vec)**(1 / 2)

        # SUPG stabilizer on/off switch;  b: float, use 0.0 or 1.0
        # To set it, e.g. `solver.enable_SUPG.b = 1.0`, where `solver` is your `AdvectionDiffusion` instance.
        enable_SUPG = Expression('b', degree=0, b=0.0)

        # [k] / ([ρ] [c]) = m² / s,  a (thermal) kinematic viscosity
        #
        # Donea & Huerta (2003, p. 65), discussing the steady-state advection-diffusion equation:
        #   "It is important to note that for higher-order [as compared to linear] finite elements, apart
        #   from the results discussed in Section 2.2.4 and Remark 2.11 [referring to analyses of the
        #   discrete equations in 1D and conditions for exact results at the nodes] (see also Franca, Frey
        #   and Hughes, 1992; Codina, 1993b), no optimal definition of `τ` exists. Numerical experiments
        #   seem to indicate that for finite elements of order `p`, the value of the stabilization parameter
        #   should be approximately `τ / p`."
        α0 = Constant(1 / self.V.ufl_element().degree())
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (k / (ρ * c)) / he**2)**-1  # seconds
        self.enable_SUPG = enable_SUPG

        # We need the strong form of the equation to compute the residual
        if self.advection == "divergence-free":
            def adv(U_):  # strong form of the modified advection operator that yields the skew-symmetric weak form
                return dot(a, nabla_grad(U_)) + (1 / 2) * div(a) * U_
        elif self.advection == "general":
            def adv(U_):
                # here the modifications cancel in the strong form
                return dot(a, nabla_grad(U_))
        else:  # self.advection == "off":
            adv = None

        # SUPG only makes sense if advection is enabled
        if adv:
            if istensor(self._k):
                def diffusion(U_):
                    return div(dot(k, nabla_grad(U_)))
            else:
                def diffusion(U_):
                    return div(k * nabla_grad(U_))

            def R(U_):
                # The residual is evaluated elementwise in strong form.
                residual = ρ * c * (dudt + adv(U_)) - diffusion(U_) - ρ * h
                if self.use_stress:
                    residual += -inner(σ, nabla_grad(a))
                return residual
            F_SUPG = enable_SUPG * τ_SUPG * dot(adv(v), R(u)) * dx
            F += F_SUPG

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
