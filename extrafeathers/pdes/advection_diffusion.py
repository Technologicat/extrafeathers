# -*- coding: utf-8; -*-
"""Advection-diffusion equation, and its physical realizations."""

__all__ = ["HeatEquation",
           "AdvectionDiffusion"]

import typing

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, DirichletBC,
                    Function, TrialFunction, TestFunction, Expression,
                    Constant, FacetNormal,
                    dot, inner, nabla_grad, div, dx, ds,
                    lhs, rhs, assemble, solve,
                    begin, end)

from ..meshfunction import meshsize, cell_mf_to_expression
from .util import ScalarOrTensor, istensor, ufl_constant_property


# TODO: Add a reaction term with a FEM function coefficient, for chemical reactions and similar.
# TODO: Make the diffusivity a FEM function. Need also a function outside the ∇· (or in the mat. der. term).
class AdvectionDiffusion:
    """Advection-diffusion equation.

      [∂u/∂t + (a·∇)u] - ∇·(ν ∇u) = τ : ∇a + g

    Time integration is performed using the θ method; Crank-Nicolson by default.

    `V`: function space for `u`
         Here `u` is the abstract generic Eulerian field that undergoes advection
         and diffusion; for example, in the heat equation, `u` is the temperature.
    `ν`: diffusivity [m² / s]
         Scalar `ν` for isotropic material; rank-2 tensor for anisotropic.
    `bc`: Dirichlet boundary conditions for temperature
    `dt`: timestep [s]
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 is Crank-Nicolson; 0 is forward Euler, 1 is backward Euler.

         Note that for θ = 0, the SUPG stabilization parameter τ_SUPG → 0,
         so when using forward Euler, it does not make sense to enable the
         SUPG stabilizer.

    `advection`: one of "off", "divergence-free", "general"
        "off":             Zero advection velocity, diffusion only.
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

    `use_stress`: Support logic, specific to the heat equation of a moving material.
                  Comes from the internal energy (enthalpy) balance of a continuum,
                  see Allen et al. (1988).

                  Whether to include the stress term  τ : ∇a,  which represents the
                  contribution of stress to internal energy. The stress is here named
                  `τ` instead of `σ` for normalization reasons; see `HeatEquation`
                  for details.

                  Due to the factor ∇a, `use_stress` can only be enabled when
                  `advection` is also enabled (i.e. something other than "off").

                  Setting `use_stress=True` is only useful if you intend to send in
                  a stress tensor (possibly at each timestep).

                  When `use_stress=False`, the stress term is discarded; runs faster.

                  When using `AdvectionDiffusion` to compute anything other than
                  heat transport in a moving material, this should be off (`False`).

    `stress_degree`: The degree of the finite element space of the stress field.
                     This must match the data you are loading in.

                     The default `None` means "use the same degree as `V`".

                     Used only when `use_stress=True`.

    The source `self.h`, the advection velocity `self.a` [m / s], and the stress tensor
    `self.σ` are assignable FEM functions. The unit of the source is [u] / s.

    For example, to set a constant source everywhere::

        h: Function = interpolate(Constant(1.0), V)
        solver.h.assign(h)

    Anything compatible with a `Function` can be assigned, including FEM fields produced
    by another solver or loaded from a file (that was produced by a solver on the same
    mesh!). Technically, `h` lives on a `FunctionSpace`, `a` on a `VectorFunctionSpace`,
    and `σ` on a `TensorFunctionSpace` compatible with `V`.

    References:
        Myron B. Allen, III, Ismael Herrera, and George F. Pinder. 1988.
        Numerical Modeling in Science and Engineering. Wiley Interscience.
    """
    def __init__(self, V: FunctionSpace,
                 ν: ScalarOrTensor,
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
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Specific heat source
        self.h = Function(V)
        self.h.vector()[:] = 0.0  # placeholder value

        # Convection velocity. FEM function for maximum generality.
        self.advection = advection
        a_degree = velocity_degree if velocity_degree is not None else V.ufl_element().degree()
        V_rank1 = VectorFunctionSpace(self.mesh, V.ufl_element().family(), a_degree)
        self.a = Function(V_rank1)
        self.a.vector()[:] = 0.0

        # Stress. FEM function for maximum generality.
        self.use_stress = use_stress and self.advection != "off"
        σ_degree = stress_degree if stress_degree is not None else V.ufl_element().degree()
        V_rank2 = TensorFunctionSpace(self.mesh, V.ufl_element().family(), σ_degree)
        self.σ = Function(V_rank2)
        self.σ.vector()[:] = 0.0

        # Parameters.
        self._ν = Constant(ν)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # SUPG stabilizer on/off switch;  b: float, use 0.0 or 1.0
        # To set it, e.g. `solver.enable_SUPG.b = 1.0`,
        # where `solver` is your `AdvectionDiffusion` instance.
        self.enable_SUPG = Expression('b', degree=0, b=0.0)

        # SUPG stabilizer tuning parameter.
        #
        # Donea & Huerta (2003, p. 65), discussing the steady-state advection-diffusion
        # equation:
        #   "It is important to note that for higher-order [as compared to linear]
        #   finite elements, apart from the results discussed in Section 2.2.4 and
        #   Remark 2.11 [referring to analyses of the discrete equations in 1D and
        #   conditions for exact results at the nodes] (see also Franca, Frey and
        #   Hughes, 1992; Codina, 1993b), no optimal definition of `τ` exists.
        #   Numerical experiments seem to indicate that for finite elements of
        #   order `p`, the value of the stabilization parameter should be
        #   approximately `τ / p`."
        self._α0 = Constant(1 / self.V.ufl_element().degree())

        self.compile_forms()

    ν = ufl_constant_property("ν", doc="Diffusivity [m² / s]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def peclet(self, u, L):
        """Return the Péclet number.

        `u`: characteristic speed (scalar) [m / s]
        `L`: length scale [m]

        The Péclet number is defined as the ratio of advective vs. diffusive
        effects::

            Pe = u L / ν

        where `ν` is the diffusivity, which has the units of kinematic viscosity.

        Choosing representative values for `u` and `L` is more of an art
        than a science. Typically:

            - In a flow past an obstacle, `u` is the free-stream speed,
              and `L` is the length of the obstacle along the direction
              of the free stream.

            - In a lid-driven cavity flow in a square cavity, `u` is the
              speed of the lid, and `L` is the length of the lid.

            - In a flow in a pipe, `u` is the free-stream speed, and `L`
              is the pipe diameter.

            - For giving the user a dynamic estimate of `Pe` during computation,
              the maximum value of `|u|` may be generally useful, but `L`
              must still be intuited from the problem geometry.
        """
        return u * L / self.ν

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # The abstract Eulerian field
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

        # Parameters
        ν = self._ν
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG = self.enable_SUPG

        # θ time integration
        U = (1 - θ) * u_n + θ * u
        dudt = (u - u_n) / dt

        # These terms are always the same:
        F = (dudt * v * dx -
             dot(h, v) * dx)

        # but the others depend on options:
        if istensor(self._ν):
            F += dot(dot(ν, nabla_grad(U)), nabla_grad(v)) * dx
            # TODO: add support for nonzero Neumann BCs
            # The full weak form of -∇·(nu·∇u) for tensor `ν` is:
            # F += (dot(dot(ν, nabla_grad(U)), nabla_grad(v)) * dx -
            #       dot(n, dot(ν, nabla_grad(U))) * v * ds)
        else:
            F += ν * dot(nabla_grad(U), nabla_grad(v)) * dx
            # TODO: add support for nonzero Neumann BCs
            # The full weak form of -∇·(nu·∇u) for scalar `ν` is:
            # F += (ν * dot(nabla_grad(U), nabla_grad(v)) * dx -
            #       ν * dot(n, nabla_grad(U)) * v * ds)

        if self.advection == "divergence-free":
            # Skew-symmetric form for divergence-free advection velocity
            # (see navier_stokes.py).
            F += ((1 / 2) * (dot(a, nabla_grad(U)) * v -
                             dot(a, nabla_grad(v)) * U) * dx +
                  (1 / 2) * dot(n, a) * U * v * ds)
        elif self.advection == "general":
            # Skew-symmetric advection, as above; but subtract the contribution from the
            # extra term (1/2) (∇·a) u, thus producing an extra symmetric term that
            # accounts for the divergence of `a`.
            F += ((1 / 2) * (dot(a, nabla_grad(U)) * v -
                             dot(a, nabla_grad(v)) * U) * dx +
                  (1 / 2) * dot(n, a) * U * v * ds -
                  (1 / 2) * div(a) * U * v * dx)
            # # Just use the asymmetric advection term as-is.
            # # TODO: which form is better? This has fewer operations, but which gives better stability?
            # F += dot(a, nabla_grad(U)) * v * dx

        if self.use_stress and self.advection != "off":
            F += -inner(σ, nabla_grad(a)) * v * dx

        # SUPG: streamline upwinding Petrov-Galerkin.
        #
        def mag(vec):
            return dot(vec, vec)**(1 / 2)
        if istensor(self._ν):
            mag_ν = inner(ν, ν)**(1 / 2)
        else:
            mag_ν = ν

        # [τ] = s
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * mag_ν / he**2)**-1

        # We need the strong form of the equation to compute the residual
        if self.advection == "divergence-free":
            # Strong form of the modified advection operator that yields the
            # skew-symmetric weak form.
            def adv(U_):
                return dot(a, nabla_grad(U_)) + (1 / 2) * div(a) * U_
        elif self.advection == "general":
            def adv(U_):
                # here the modifications cancel in the strong form
                return dot(a, nabla_grad(U_))
        else:  # self.advection == "off":
            adv = None

        # SUPG only makes sense if advection is enabled
        if adv:
            if istensor(self._ν):
                def diffusion(U_):
                    return div(dot(ν, nabla_grad(U_)))
            else:
                def diffusion(U_):
                    return div(ν * nabla_grad(U_))

            def R(U_):
                # The residual is evaluated elementwise in strong form.
                residual = (dudt + adv(U_)) - diffusion(U_) - h
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


class HeatEquation(AdvectionDiffusion):
    """Heat transport, optionally in a moving material.

    Like `AdvectionDiffusion`, but instead of `ν`, we have three parameters:

    `ρ`: density [kg / m³]
    `c`: specific heat capacity [J / (kg K)]
    `k`: heat conductivity [W / (m K)]
         Scalar `k` for thermally isotropic material; rank-2 tensor for anisotropic.

    The equation for the absolute temperature `T`,  [T] = K,  is based on the
    internal energy (enthalpy) balance of a continuum (Allen et al., 1988):

      ρ c [∂T/∂t + (a·∇)T] - ∇·(k ∇T) = σ : ∇a + ρ h

    where

      - `T` is the absolute temperature [K],
      - `a` is the advection velocity [m / s], and
      - `σ` is the stress tensor [Pa].

    Each term has the units of a volumetric power, [W / m³].

    This form of the equation is based on Fourier's law, and the modeling
    assumption that the specific internal energy `e` [J / kg] behaves as

      e = c T

    i.e. there are no phase changes.

    Strictly, we also assume that `c` is a constant independent of the
    temperature `T`. If  c = c(T),  we must add the term

      ρ ∂c/∂T T [∂T/∂t + (a·∇)T]

    to the LHS, and the equation becomes nonlinear; this is currently
    not supported by this solver.


    **Normalization**

    The equation is actually solved as normalized by the volumetric heat capacity
    `ρ c` [J / (m³ K)], in the form

      [∂u/∂t + (a·∇)u] - ∇·(ν ∇u) = τ : ∇a + g

    where  ν = k / (ρ c)  is the diffusivity, which has the units of a kinematic
    viscosity:  [ν] = m² / s.  Each term has the units of [u] / s. Because for
    the heat equation,  u = T,  concretely the units are K / s.

    Due to this normalization, instead of the raw stress field `σ` [Pa] and a
    specific source `h` [W / kg] (as in the raw equation), the solver actually
    expects to get as its `σ` and `h` the following quantities:

        τ = σ / (ρ c)
        g = h / c

    Thus, if you wish to set the stress and source fields `σ` and `h`, normalize
    your input data accordingly::

        solver.σ = raw_stress / (solver.ρ * solver.c)
        solver.h = specific_source / solver.c

    where `solver` is a `HeatEquation` instance.

    The diffusivity `ν` is automatically computed, and available as `self.ν`.
    The type is the same as that of `k` (`float` or a rank-2 tensor).
    Writing to `ρ`, `c`, or `k` automatically updates the diffusivity.

    References:
        Myron B. Allen, III, Ismael Herrera, and George F. Pinder. 1988.
        Numerical Modeling in Science and Engineering. Wiley Interscience.
    """
    def __init__(self, V: FunctionSpace,
                 ρ: float, c: float, k: ScalarOrTensor,
                 bc: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5, *,
                 advection: str = "divergence-free",
                 velocity_degree: int = None,
                 use_stress: bool = False,
                 stress_degree: int = None):
        # Here we don't need to wrap the parameters into UFL `Constant` objects,
        # but we need to make them properties that update the underlying `ν`
        # (in our parent `AdvectionDiffusion`) when written to.
        self._ρ = ρ
        self._c = c
        self._k = k
        super().__init__(V, self._update_ν(), bc, dt, θ,
                         advection=advection,
                         velocity_degree=velocity_degree,
                         use_stress=use_stress,
                         stress_degree=stress_degree)

    def _update_ν(self):
        """Compute the diffusivity ν from ρ, c, and k."""
        self._ν = self.k / (self.ρ * self.c)
        return self._ν

    def _get_ρ(self):
        return self._ρ
    def _set_ρ(self, ρ):
        self._ρ = ρ
        self._update_ν()
    ρ = property(fget=_get_ρ, fset=_set_ρ, doc="Density [kg / m³]")

    def _get_c(self):
        return self._c
    def _set_c(self, c):
        self._c = c
        self._update_ν()
    c = property(fget=_get_c, fset=_set_c, doc="Specific heat capacity [J / (kg K)]")

    def _get_k(self):
        return self._k
    def _set_k(self, k):
        self._k = k
        self._update_ν()
    k = property(fget=_get_k, fset=_set_k, doc="Heat conductivity [W / (m K)]")
