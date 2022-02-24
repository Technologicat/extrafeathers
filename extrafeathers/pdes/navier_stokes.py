# -*- coding: utf-8; -*-
"""
Incompressible Navier-Stokes equations:

  ρ (∂u/∂t + u·∇u) - ∇·σ(u, p) = ρ f
                            ∇·u = 0

where

  σ = 2 μ ε(u) - p I
  ε = symm ∇u

and I is the rank-2 identity tensor.
"""

__all__ = ["NavierStokes"]

import typing

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Function, TrialFunction, TestFunction, Expression,
                    Constant, FacetNormal,
                    dot, inner, sym,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve, project,
                    begin, end)

from ..meshfunction import meshsize, cell_mf_to_expression
from .util import ufl_constant_property


def ε(u):
    """Symmetric gradient of the velocity field `u`.

        (symm ∇)(u) = (1/2) (∇u + transpose(∇u))

    Despite the name "ε", this is the *strain rate*  dε/dt,
    where `d/dt` denotes the *material derivative*.

    This method returns a UFL expression for the whole symmetric gradient.
    If you want to plot, extract and interpolate or project what you need.
    For example, to plot ε11::

        from fenics import interpolate, plot

        # scalar function space
        W = V.sub(0).collapse()
        # W = FunctionSpace(mesh, 'P', 2)  # can do this, too

        ε = ε(solver.u_)
        ε11 = ε.sub(0)
        plot(interpolate(ε11, W))

    Note in FEniCS the tensor components are indexed linearly,
    storage is by row. E.g. in 2D::

        ε11 = ε.sub(0)
        ε12 = ε.sub(1)
        ε21 = ε.sub(2)
        ε22 = ε.sub(3)

    See:
        https://fenicsproject.org/qa/4458/how-can-i-get-two-components-of-a-tensorfunction/
    """
    return sym(nabla_grad(u))

def σ(u, p, μ):
    """Stress tensor of isotropic Newtonian fluid.

        σ = 2 μ (symm ∇)(u) - p I

    This method returns a UFL expression the whole stress tensor. If you want
    to plot, extract and interpolate or project what you need. For example,
    to plot the von Mises stress::

        from dolfin import tr, Identity, sqrt, inner
        from fenics import project, plot

        # scalar function space
        W = V.sub(0).collapse()  # use the space of the first comp. of `V`
        # W = FunctionSpace(mesh, 'P', 2)  # or create your own space

        def dev(T):
            '''Deviatoric part of rank-2 tensor `T`.'''
            return T - (1 / 3) * tr(T) * Identity(T.geometric_dimension())
        # `solver._μ` is the UFL `Constant` object; `solver.`
        σ = σ(solver.u, solver.p, solver._μ)
        s = dev(σ)
        vonMises = sqrt(3 / 2 * inner(s, s))
        plot(project(vonMises, W))
    """
    return 2 * μ * ε(u) - p * Identity(p.geometric_dimension())


# TODO: use nondimensional form
class NavierStokes:
    """Incompressible Navier-Stokes solver, no turbulence model.

    Main use case is laminar flow. Can be used for direct numerical simulation (DNS)
    of turbulent flows, if the mesh is fine enough.

    Uses the IPCS method (incremental pressure correction scheme; Goda, 1979),
    as presented in the FEniCS tutorial. Stabilized with SUPG, LSIC, and
    skew-symmetric advection. See Donea & Huerta (2003).

    Time integration is performed using the θ method; Crank-Nicolson by default.

    `V`: function space for velocity
    `Q`: function space for pressure
    `ρ`: density [kg / m³]
    `μ`: dynamic viscosity [Pa s]
    `bcu`: Dirichlet boundary conditions for velocity
    `bcp`: Dirichlet boundary conditions for pressure
    `dt`: timestep [s]
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 is Crank-Nicolson; 0 is forward Euler, 1 is backward Euler.

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.
    The function spaces `V` and `Q` must be LBB-compatible. If unsure, use Taylor-Hood
    (a.k.a. P2P1) elements: choose `P2` for `V` and `P1` for `Q`. Q2Q1 works, too.

    The specific body force `self.f` [N / kg] = [m / s²] is an assignable FEM function.
    Interpolate or project what you want onto `V`, and then `.assign`. For example,
    to set a constant body force everywhere::

        f: Function = interpolate(Constant((0, -9.81)), V)
        solver.f.assign(f)

    (Note that if you do simulate gravity, you may need to change some of your boundary
    conditions to accommodate.)

    References:
        Jean Donea and Antonio Huerta. 2003. Finite Element Methods for Flow Problems.
        Wiley. ISBN 0-471-49666-9.

        Katuhiko Goda. A multistep technique with implicit difference schemes for
        calculating two- or three-dimensional cavity flows. Journal of Computational
        Physics, 30(1):76–95, 1979.
    """
    def __init__(self, V: VectorFunctionSpace, Q: FunctionSpace,
                 ρ: float, μ: float,
                 bcu: typing.List[DirichletBC],
                 bcp: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        self.V = V
        self.Q = Q
        self.bcu = bcu
        self.bcp = bcp

        # Function space of ℝ (single global DOF), for computing the average pressure
        self.W = FunctionSpace(self.mesh, "R", 0)

        # Trial and test functions
        self.u = TrialFunction(V)  # no suffix: the UFL symbol for the unknown quantity
        self.v = TestFunction(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)

        # Functions for solution at previous and current time steps
        self.u_n = Function(V)  # suffix _n: the old value (end of previous timestep)
        self.u_ = Function(V)  # suffix _: the latest computed approximation
        self.p_n = Function(Q)
        self.p_ = Function(Q)

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Specific body force. FEM function for maximum generality.
        self.f = Function(V)
        self.f.vector()[:] = 0.0  # placeholder value

        # Parameters.
        self._ρ = Constant(ρ)
        self._μ = Constant(μ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)
        self._α0 = Constant(1)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    μ = ufl_constant_property("μ", doc="Dynamic viscosity [Pa s]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def reynolds(self, uinf, L):
        """Return the Reynolds number of the flow.

        `uinf`: free-stream speed [m / s]
        `L`: length scale [m]
        """
        nu = self.μ / self.ρ  # kinematic viscosity,  [ν] = m² / s
        return uinf * L / nu

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Velocity
        u = self.u      # new (unknown)
        v = self.v      # test
        u_ = self.u_    # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Pressure
        p = self.p
        q = self.q
        p_ = self.p_
        p_n = self.p_n

        # Specific body force
        f = self.f

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        μ = self._μ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        # Define variational problem for step 1 (tentative velocity)
        #
        # This is just the variational form of the momentum equation.
        #
        #
        # **Stress term**
        #
        # The boundary terms must match the implementation of the σ term. Note we use the
        # old pressure p_n and the midpoint value of velocity U.
        #
        # Defining
        #   U = (1 - θ) u_n + θ u
        # we have
        #   σ(U, p_n) = 2 μ * ε(U) - p_n * I = 2 μ * ε( (1 - θ) u_n + θ u ) - p_n * I
        # where I is the rank-2 identity tensor.
        #
        # Integrating -(∇·σ(U))·v dx by parts, we obtain
        #   σ(U) : ∇v dx = σ(U) : symm∇(v) dx = σ(U) : ε(v)
        # (we can use symm∇, because σ is symmetric) plus the boundary term
        #   -[σ(U) · n] · v ds = -2 μ [ε(U) · n] · v ds  +  p_n [n · v] ds
        #
        # Then requiring ∂u/∂n = 0 on the Neumann boundary (for fully developed
        # outflow) eliminates one of the terms from inside the ε in the
        # boundary term, leaving just the other one, and the pressure term.
        # Here we use the transpose jacobian convention for the gradient of
        # a vector, (∇u)ik := ∂i uk, so the term that remains after setting
        # n · ∇u = ni ∂i uk = 0 is (∂i uk) nk (which comes from the transposed
        # part of the symm∇). This produces the term -μ [[∇U] · n] · v ds.
        #
        #
        # **Convective term**
        #
        # In the discretization, the convection velocity field and the velocity
        # field the convection operator is applied to are allowed to be
        # different, so let us call the convection velocity field `a`.
        #
        # The straightforward weak form is
        #   a · ∇u · v dx
        #
        # Using the old value for both a and u, i.e. a = u = u_n, gives us an
        # explicit scheme for the convection.
        #
        # A semi-implicit scheme for convection is obtained when a = u_n, and u
        # is the unknown, end-of-timestep value of the step 1 velocity
        # (sometimes called the intermediate velocity).
        #
        # Donea & Huerta (2003, sec. 6.7.1) remark that in the 2000s, it has
        # become standard to use this skew-symmetric weak form:
        #   (1/2) a · [∇u · v - ∇v · u] dx
        # which in the strong form is equivalent with replacing the convective term
        #   (a·∇) u
        # by the modified term
        #   (a·∇) u  +  (1/2) (∇·a) u
        # This is consistent for an incompressible flow, and necessary for
        # unconditional time stability for schemes that are able to provide it.
        #
        # To see this equivalence, consider the conversion of the modified term
        # into weak form:
        #    (a·∇) u · v dx  +  (1/2) (∇·a) u · v dx
        # Observing that
        #    ∂i (ai uk vk) = (∂i ai) uk vk + ai ∂i (uk vk)
        #    ∇·(a (u · v)) = (∇·a) u · v  +  a · ∇(u · v)
        # we use the divergence theorem in the last term of the weak form, obtaining
        #    (a·∇) u · v dx  -  (1/2) a · ∇(u · v) dx  +  (1/2) n · a (u · v) ds
        # Furthermore, noting that
        #    a · ∇(u · v) = ai ∂i (uk vk)
        #                  = ai (∂i uk) vk + ai uk (∂i vk)
        #                  = a · ∇u · v  +  a · ∇v · u
        #                  = a · [∇u · v + ∇v · u]
        # and
        #    (a·∇) u · v = ((ai ∂i) uk) vk = ai (∂i uk) vk = a · ∇u · v
        # we have the terms
        #      a · ∇u · v dx
        #    - (1/2) a · [∇u · v + ∇v · u] dx
        #    + (1/2) n · a (u · v) ds
        # Cleaning up, we obtain
        #    (1/2) a · [∇u · v - ∇v · u] dx  +  (1/2) n · a (u · v) ds
        # as claimed. Keep in mind the boundary term, which contributes on boundaries
        # through which there is flow (i.e. inlets and outlets) - we do not want to
        # introduce an extra boundary condition.
        #
        # Another way to view the role of the extra term in the skew-symmetric form is to
        # consider the Helmholtz decomposition of the convection velocity `a`:
        #   a = ∇φ + ∇×A
        # where φ is a scalar potential (for the irrotational part) and A is a
        # vector potential (for the divergence-free part). We have
        #   (∇·a) = ∇·∇φ + ∇·∇×A = ∇²φ + 0
        # so the extra term is proportional to the laplacian of the scalar potential:
        #   (∇·a) u = (∇²φ) u
        #
        # References:
        #     Jean Donea and Antonio Huerta. 2003. Finite Element Methods
        #     for Flow Problems. Wiley. ISBN 0-471-49666-9.
        #
        U = (1 - θ) * u_n + θ * u
        dudt = (u - u_n) / dt
        # # Original convective term
        # F1 = (ρ * dot(dudt, v) * dx +
        #       ρ * dot(dot(u_n, nabla_grad(u_n)), v) * dx +
        #       inner(σ(U, p_n), ε(v)) * dx +
        #       dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
        #       dot(f, v) * dx)
        #
        # Skew-symmetric convective term
        a = u_n  # convection velocity
        ustar = U  # quantity the convection operator applies to
        # F1 = (ρ * dot(dudt, v) * dx +
        #       ρ * (1 / 2) * (dot(dot(a, nabla_grad(ustar)), v) -
        #                      dot(dot(a, nabla_grad(v)), ustar)) * dx +
        #       ρ * (1 / 2) * dot(n, a) * dot(ustar, v) * ds +
        #       inner(σ(U, p_n), ε(v)) * dx +
        #       dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
        #       dot(f, v) * dx)
        #
        # Split, so we can re-assemble only the part that changes at each timestep.
        # Keep in mind:
        #  - U = (1 - θ) u_n +  θ u.  Anything times `u` goes to LHS,
        #    while the rest goes to RHS.
        #  - In the convection term, we have the product of `u_n` and `u`.
        #    This is an LHS term that depends on time-varying data.
        #  - In the stress term, `u` and `p_n` appear in different terms of the sum.
        #
        # Units of the weak form?
        #   [ρ ∂u/∂t v dx] = (kg / m³) * ((m / s) / s) * (m / s) * m³
        #                  = (kg m / s²) m / s
        #                  = N m / s
        #                  = J / s
        #                  = W
        # so each term in the weak form momentum equation (in 3D) is a virtual power.
        F1_varying = (ρ * (1 / 2) * (dot(dot(a, nabla_grad(ustar)), v) -
                                     dot(dot(a, nabla_grad(v)), ustar)) * dx +
                      ρ * (1 / 2) * dot(n, a) * dot(ustar, v) * ds)
        F1_constant = (ρ * dot(dudt, v) * dx +
                       inner(σ(U, p_n, μ), ε(v)) * dx +
                       dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
                       ρ * dot(f, v) * dx)

        # LSIC: Artificial diffusion for least-squares stabilization on the
        # incompressibility constraint.
        #
        # Consistent, since for the true solution div(u) = 0. Provides
        # additional stability at high Reynolds numbers.
        #
        # See Donea & Huerta (2003, sec. 6.7.2, pp. 296-297).
        #
        # Interpretation?
        #   ∂k ((∂i ui) vk) = (∂i ui) (∂k vk) + (∂k ∂i ui) vk
        # which is to say
        #   ∇·((∇·u) v) = (∇·u) (∇·v) + v · (∇ (∇·u))
        # so integrating by parts (divergence theorem)
        #   ∫ (∇·u) (∇·v) dΩ = ∫ n·((∇·u) v) dΓ - ∫ v · (∇ (∇·u)) dΩ
        # so this is in effect penalizing `grad(div(u))` in the direction of `v`.
        #
        # Note [grad(div(u))] = (1 / m²) (m / s) = 1 / (m s)
        # so [τ_LSIC] [grad(div(u))] = (m² / s) (1 / (m s)) = m / s²,
        # an acceleration; thus matching the other terms of the strong form.
        #
        # Note we use the unknown `u`, not the `U` used in the θ method; only
        # the solution at the end of the timestep should be penalized for
        # deviation from known properties of the true solution.
        def mag(vec):
            return dot(vec, vec)**(1 / 2)

        # LSIC stabilizer on/off switch;  b: float, use 0.0 or 1.0
        # To set it, e.g. `solver.enable_LSIC.b = 1.0`,
        # where `solver` is your `NavierStokes` instance.
        enable_LSIC = Expression('b', degree=0, b=1.0)
        self.enable_LSIC = enable_LSIC

        τ_LSIC = mag(a) * he / 2  # [τ_LSIC] = (m / s) * m = m² / s, a kinematic viscosity
        F1_LSIC = enable_LSIC * τ_LSIC * div(u) * div(v) * dx
        F1_varying += F1_LSIC

        # SUPG: streamline upwinding Petrov-Galerkin.
        #
        # Residual-based, consistent. Stabilizes the Galerkin formulation in the presence
        # of a dominating convective term in the momentum equation.
        #
        # See Donea & Huerta (2003), sec. 6.7.2 (p. 296), 6.5.8 (p. 287),
        # and 5.4.6.2 (p. 232). See also sec. 2.4, p. 59 ff.

        # SUPG stabilizer on/off switch;  b: float, use 0.0 or 1.0
        # To set it, e.g. `solver.enable_SUPG.b = 1.0`,
        # where `solver` is your `NavierStokes` instance.
        enable_SUPG = Expression('b', degree=0, b=1.0)
        self.enable_SUPG = enable_SUPG

        # [μ] / [ρ] = (Pa s) / (kg / m³)
        #           = (N s / m²) / (kg / m³)
        #           = (kg m s / (m² s²)) / (kg / m³)
        #           = (kg / (m s)) * m³ / kg
        #           = m² / s,  kinematic viscosity
        # Donea & Huerta (2003, p. 232), based on Shakib et al. (1991).
        # Donea & Huerta, p. 288: "α₀ = 1 / 3 appears to be optimal for linear elements"
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (μ / ρ) / he**2)**-1  # [τ] = s
        # Strong form of the modified advection operator that yields the
        # skew-symmetric weak form.
        def adv(U_):
            # To be consistent, we use the same advection velocity `a` as the
            # step 1 equation itself.
            return dot(a, nabla_grad(U_)) + (1 / 2) * div(a) * U_
        def R(U_, P_):
            # The residual is evaluated elementwise in strong form.
            return ρ * ((U_ - u_n) / dt + adv(U_)) - div(σ(U_, P_, μ)) - ρ * f
        F_SUPG = enable_SUPG * τ_SUPG * dot(adv(v), R(u, p_n)) * dx
        F1_varying += F_SUPG

        self.a1_varying = lhs(F1_varying)
        self.a1_constant = lhs(F1_constant)
        self.L1 = rhs(F1_varying + F1_constant)  # RHS is reassembled at every timestep

        # Define variational problem for step 2 (pressure correction)
        #
        # Subtract the momentum equation, written in terms of tentative
        # velocity u_ and old pressure p_n, from the momentum equation written
        # in terms of new unknown velocity u and new unknown pressure p.
        #
        # The momentum equation is
        #
        #   ρ ( ∂u/∂t + u·∇u ) = ∇·σ + ρ f
        #                       = ∇·(μ symm∇u - p I) + ρ f
        #                       = ∇·(μ symm∇u) - ∇p + ρ f
        # so we have
        #
        #   ρ ( ∂(u - u_)/∂t + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Discretizing the time derivative yields
        #
        #   ρ ( (u - u_n)/dt - (u_ - u_n)/dt + (u - u_)·∇(u - u_) )
        # = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Canceling the u_n,
        #
        #   ρ ( (u - u_)/dt + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Rearranging,
        #
        #   ρ (u - u_)/dt + ∇(p - p_n) = ∇·(μ symm∇(u - u_)) - ρ (u - u_)·∇(u - u_)
        #
        # Now, if u_ is "close enough" to u, we may take the RHS to be zero (Goda, 1979;
        # see also e.g. Landet and Mortensen, 2019, section 3).
        #
        # The result is the velocity correction equation, which we will use in
        # step 3 below:
        #
        #   ρ (u - u_) / dt + ∇p - ∇p_n = 0
        #
        # For step 2, take the divergence of the velocity correction equation,
        # and use the continuity equation to eliminate div(u) (it must be zero
        # for the new unknown velocity); obtain a Poisson problem for the new
        # pressure p, in terms of the old pressure p_n and the tentative
        # velocity u_:
        #
        #   -ρ ∇·u_ / dt + ∇²p - ∇²p_n = 0
        #
        # See also Langtangen and Logg (2016, section 3.4).
        #
        #   Katuhiko Goda. A multistep technique with implicit difference
        #       schemes for calculating two- or three-dimensional cavity flows.
        #       Journal of Computational Physics, 30(1):76–95, 1979.
        #
        #   Tormod Landet, Mikael Mortensen, 2019. On exactly incompressible DG
        #       FEM pressure splitting schemes for the Navier-Stokes equation.
        #       arXiv: 1903.11943v1
        #
        #   Hans Petter Langtangen, Anders Logg (2016). Solving PDEs in Python:
        #       The FEniCS Tutorial 1. Simula Springer Briefs on Computing 3.

        # When there are no Dirichlet BCs on p, we just rely on the fact that
        # the Krylov solvers used by FEniCS can handle singular systems
        # natively. We will postprocess to zero out the average pressure (to
        # make the pressure always unique).
        #
        # https://fenicsproject.org/qa/2406/solve-poisson-problem-with-neumann-bc/
        #
        # Using a Lagrange multiplier is another option:
        # https://fenicsproject.org/olddocs/dolfin/latest/python/demos/neumann-poisson/demo_neumann-poisson.py.html
        #
        # (To see where the weak forms for the Lagrange multiplier term come
        # from, consider the Lagrangian associated with the PDE; add the term
        # λ * <constraint>; multiply by test function; integrate over Ω; take the
        # Fréchet derivatives (or alternatively, Gateaux derivatives) w.r.t.
        # `u` and `λ` and sum them to get the Euler-Lagrange equation; group
        # terms by test function.)
        #
        # Here the LHS coefficients are constant in time.
        self.a2_constant = dot(nabla_grad(p), nabla_grad(q)) * dx
        self.L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (ρ / dt) * div(u_) * q * dx

        # PSPG: pressure-stabilizing Petrov-Galerkin.
        #
        # Used together with SUPG, for some schemes this allows the use of
        # LBB-incompatible elements; but unfortunately not for IPCS (Donea and
        # Huerta, 2003, remark 6.22, pp. 303-304).
        #
        # We will still apply PSPG in hopes that it helps stability at high Re;
        # at least we then treat both equations consistently.
        #
        # Consistent, residual-based.
        #
        # There was no value for τ_PSPG given in the book, so we recycle τ_SUPG here.
        # There was a reference, though, to a paper by Tezduyar & Osawa (2000),
        # which discusses element-local stabilization parameter values in more detail;
        # I'll need to check that out later (see p. 297 in the book for the reference).
        τ_PSPG, enable_PSPG = τ_SUPG, enable_SUPG
        F_PSPG = enable_PSPG * τ_PSPG * dot(nabla_grad(q), R(u_, p)) * dx
        self.a2_varying = lhs(F_PSPG)
        self.L2 += rhs(F_PSPG)

        # Define variational problem for step 3 (velocity correction)
        # Here the LHS coefficients are constant in time.
        self.a3 = ρ * dot(u, v) * dx
        self.L3 = ρ * dot(u_, v) * dx - dt * dot(nabla_grad(p_ - p_n), v) * dx

        # Assemble matrices (constant in time; do this once at the start)
        self.A1_constant = assemble(self.a1_constant)
        self.A2_constant = assemble(self.a2_constant)
        self.A3 = assemble(self.a3)

    def step(self) -> None:
        """Take a timestep of length `self.dt`.

        Updates `self.u_` and `self.p_`.
        """
        # Step 1: Tentative velocity step
        begin("Tentative velocity")
        A1 = self.A1_constant + assemble(self.a1_varying)
        b1 = assemble(self.L1)
        [bc.apply(A1) for bc in self.bcu]
        [bc.apply(b1) for bc in self.bcu]
        solve(A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg')
        end()

        # Step 2: Pressure correction step
        begin("Pressure correction")
        A2 = self.A2_constant + assemble(self.a2_varying)
        b2 = assemble(self.L2)
        [bc.apply(A2) for bc in self.bcp]
        [bc.apply(b2) for bc in self.bcp]
        solve(A2, self.p_.vector(), b2, 'bicgstab', 'hypre_amg')
        end()

        # Step 2½: Zero out the average pressure.
        #
        # Shifting `p` by a constant does not matter as far as the momentum
        # equation is concerned. We L2-project `p` onto ℝ, and then subtract
        # the result from `p_`, hence making its mean zero.
        #
        # This makes for a nicer-looking visualization, as well as defines the
        # pressure uniquely even in cases where we have only Neumann BCs on pressure
        # (e.g. standard cavity flow test cases).
        #
        # How to extract the single value of a `Function` on Reals (single
        # global DOF) is not documented very well. See the source code of
        # `dolfin.Function`, it has a `__float__` method. The sanity checks
        # indicate it is intended precisely for this.
        avgp = project(self.p_, self.W)
        self.p_.vector()[:] -= float(avgp)

        # Step 3: Velocity correction step
        begin("Velocity correction")
        b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), b3, 'cg', 'sor')
        end()

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This copies `self.u_` to `self.u_n` and `self.p_` to `self.p_n`,
        making the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.
        """
        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)