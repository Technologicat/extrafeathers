# -*- coding: utf-8; -*-
"""
Originally based on the FEniCS tutorial demo program:
Incompressible Navier-Stokes equations for flow around a cylinder
using the Incremental Pressure Correction Scheme (IPCS).

  ρ (∂u/∂t + u·∇u) - ∇·σ(u, p) = ρ f
                            ∇·u = 0

where

  σ = 2 μ ε(u) - p I
  ε = symm ∇u

and I is the rank-2 identity tensor.

Based on the customized version in `extrafeathers`,
packaged into a class so that we can easily use this
as a component in a coupled problem.
"""

__all__ = ["LaminarFlow"]

import typing

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Function, TrialFunction, TestFunction,
                    Constant, FacetNormal,
                    dot, inner, sym,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve,
                    begin, end)

from extrafeathers import autoboundary


# TODO: Add support for resuming (use TimeSeries to store u_ and p_?)

# TODO: Initialize u_ and p_ from a potential-flow approximation to have a physically reasonable initial state.
# TODO: We need to implement a potential-flow solver to be able to do that. It's just a Poisson equation,
# TODO: but the scalings must match this solver, and we need to produce a pressure field, too.

def ε(u):
    """Symmetric gradient of the velocity field `u`.

    Despite the name, this is the *strain rate*  dε/dt,
    where `d/dt` denotes the *material derivative*.

    This method returns the whole symmetric gradient. If you want to plot,
    you'll have to extract and interpolate what you need.

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

    This method returns the whole stress tensor. If you want to plot,
    you'll have to extract and interpolate what you need.

    For example, to plot the von Mises stress::

        from dolfin import tr, Identity, sqrt, inner
        from fenics import interpolate, plot

        # scalar function space
        W = V.sub(0).collapse()
        # W = FunctionSpace(mesh, 'P', 2)  # can do this, too

        def dev(T):
            '''Deviatoric part of rank-2 tensor `T`.'''
            return T - (1 / 3) * tr(T) * Identity(T.geometric_dimension())
        σ = σ(solver.u, solver.p, solver.μ)
        s = dev(σ)
        vonMises = sqrt(3 / 2 * inner(s, s))
        plot(interpolate(vonMises, W))
    """
    return 2 * μ * ε(u) - p * Identity(p.geometric_dimension())


class LaminarFlow:
    """Laminar flow (incompressible Navier-Stokes) solver based on the FEniCS tutorial example.

    Uses the IPCS method (incremental pressure correction scheme; Goda 1979).

    `V`: function space for velocity
    `Q`: function space for pressure
    `ρ`: density [kg / m³]
    `μ`: dynamic viscosity [Pa s]
    `dt`: timestep [s]
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 gives the implicit midpoint rule; 0 is forward Euler,
         and 1 is backward Euler.

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.

    The specific body force `self.f` [N / kg] = [m / s²] is an assignable FEM function.
    For example, to set a constant body force everywhere::

        f: Function = interpolate(Constant((0, -9.81)), V)
        solver.f.assign(f)

    (Note that if you do simulate gravity, you may need to change some of your boundary
    conditions to accommodate.)
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
        self.h = autoboundary.cell_meshfunction_to_expression(autoboundary.meshsize(self.mesh))

        # Specific body force
        self.f = Function(V)
        self.f.vector()[:] = 0.0  # placeholder value

        # Parameters.
        #
        # We must initialize the underlying variables for properties directly
        # to avoid triggering the compile before all necessary parameters are
        # initialized.
        self._ρ = ρ
        self._μ = μ
        self._dt = dt
        self._θ = θ
        self.compile_forms()

    def _set_ρ(self, ρ: float) -> None:
        self._ρ = ρ
        self.compile_forms()
    def _get_ρ(self) -> float:
        return self._ρ
    ρ = property(fget=_get_ρ, fset=_set_ρ, doc="Density [kg / m³]")

    def _set_μ(self, μ: float) -> None:
        self._μ = μ
        self.compile_forms()
    def _get_μ(self) -> float:
        return self._μ
    μ = property(fget=_get_μ, fset=_set_μ, doc="Dynamic viscosity [Pa s]")

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

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Velocity
        u = self.u  # new (unknown)
        v = self.v  # test
        u_ = self.u_  # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Pressure
        p = self.p
        q = self.q  # test
        p_ = self.p_
        p_n = self.p_n

        # Specific body force
        f = self.f

        # Local mesh size (for stabilization terms)
        h = self.h

        # Wrap constant parameters in a Constant to allow changing the value without triggering a recompile
        ρ = Constant(self.ρ)
        μ = Constant(self.μ)
        dt = Constant(self.dt)
        θ = Constant(self.θ)

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
        # Integrating -(∇·σ(U))·v dx by parts, we get σ(U) : ∇v dx = σ(U) : symm∇(v) dx = σ(U) : ε(v)
        # (we can use symm∇, because σ is symmetric) plus the boundary term
        #   -[σ(U) · n] · v ds = -2 μ [ε(U) · n] · v ds  +  p_n [n · v] ds
        #
        # Then requiring ∂u/∂n = 0 on the Neumann boundary (for fully developed outflow) eliminates one
        # of the terms from inside the ε in the boundary term, leaving just the other one, and the
        # pressure term. Here we use the transpose jacobian convention for the gradient of a vector,
        # (∇u)ik := ∂i uk, so the term that remains after setting  n · ∇u = ni ∂i uk = 0  is
        # (∂i uk) nk  (which comes from the transposed part of the symm∇). This produces the term
        # -μ [[∇U] · n] · v ds.
        #
        #
        # **Convective term**
        #
        # In the discretization, the convection velocity field and the velocity field the
        # convection operator is applied to are allowed to be different, so let us call the
        # convection velocity field `a`.
        #
        # The straightforward weak form is
        #   a · ∇u · v dx
        #
        # Using the old value for both a and u, i.e.  a = u = u_n,  gives us an explicit scheme
        # for the convection.
        #
        # A semi-implicit scheme for convection is obtained when  a = u_n, and u is the unknown,
        # end-of-timestep value of the step 1 velocity (sometimes called the intermediate velocity).
        #
        # Donea & Huerta (2003, sec. 6.7.1) remark that in the 2000s, it has become standard to
        # use this skew-symmetric weak form:
        #   (1/2) a · [∇u · v - ∇v · u] dx
        # which in the strong form is equivalent with replacing the convective term
        #   (a·∇) u
        # by the modified term
        #   (a·∇) u  +  (1/2) (∇·a) u
        # This is consistent for an incompressible flow, and necessary for unconditional time stability.
        #
        # To see this equivalence, consider the conversion of the modified term into weak form:
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
        # we have
        #    a · ∇u · v dx  -  (1/2) a · [∇u · v + ∇v · u] dx  +  (1/2) n · a (u · v) ds
        # Finally, cleaning up, we obtain
        #    (1/2) a · [∇u · v - ∇v · u] dx  +  (1/2) n · a (u · v) ds
        # as claimed. Keep in mind the boundary term, which contributes on boundaries
        # through which there is flow (i.e. inlets and outlets) - we do not want to
        # introduce an extra boundary condition.
        #
        # Another way to view the role of the extra term in the skew-symmetric form is to
        # consider the Helmholtz decomposition of the convection velocity `a`:
        #   a = ∇φ + ∇×A
        # where φ is a scalar potential (for the irrotational part) and A is a vector potential
        # (for the divergence-free part). We have
        #   (∇·a) = ∇·∇φ + ∇·∇×A = ∇²φ + 0
        # so the extra term is proportional to the laplacian of the scalar potential:
        #   (∇·a) u = (∇²φ) u
        #
        # References:
        #     Jean Donea and Antonio Huerta. 2003. Finite Element Methods for Flow Problems.
        #     Wiley. ISBN 0-471-49666-9.
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
        #       ρ * (1 / 2) * (dot(dot(a, nabla_grad(ustar)), v) - dot(dot(a, nabla_grad(v)), ustar)) * dx +
        #       ρ * (1 / 2) * dot(n, a) * dot(ustar, v) * ds +
        #       inner(σ(U, p_n), ε(v)) * dx +
        #       dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
        #       dot(f, v) * dx)
        #
        # Split, so we can re-assemble only the part that changes at each timestep.
        # Keep in mind:
        #  - U = (1 - θ) u_n +  θ u.  Anything times `u` goes to LHS, while the rest goes to RHS.
        #  - In the convection term, we have the product of `u_n` and `u`. This is an LHS term that
        #    depends on time-varying data.
        #  - In the stress term, `u` and `p_n` appear in different terms of the sum.
        F1_varying = (ρ * (1 / 2) * (dot(dot(a, nabla_grad(ustar)), v) -
                                     dot(dot(a, nabla_grad(v)), ustar)) * dx +
                      ρ * (1 / 2) * dot(n, a) * dot(ustar, v) * ds)
        F1_constant = (ρ * dot(dudt, v) * dx +
                       inner(σ(U, p_n, μ), ε(v)) * dx +
                       dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
                       ρ * dot(f, v) * dx)

        # LSIC: Artificial diffusion for least-squares stabilization on the incompressibility constraint.
        # Helps at high Reynolds numbers. See Donea & Huerta (2003, sec. 6.7.2).
        τ_LSIC = (dot(u_n, u_n))**(1 / 2) * h / 2  # [m² / s], like a kinematic viscosity
        F1_varying += τ_LSIC * div(U) * div(v) * dx

        self.a1_varying = lhs(F1_varying)
        self.a1_constant = lhs(F1_constant)
        self.L1 = rhs(F1_varying + F1_constant)  # RHS is reassembled at every timestep anyway

        # Define variational problem for step 2 (pressure correction)
        #
        # Subtract the momentum equation, written in terms of tentative velocity u_ and old pressure p_n,
        # from the momentum equation written in terms of new unknown velocity u and new unknown pressure p.
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
        # Discretizing the time derivative,
        #
        #   ρ ( (u - u_n)/dt - (u_ - u_n)/dt + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Canceling the u_n,
        #
        #   ρ ( (u - u_)/dt + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Rearranging,
        #   ρ (u - u_)/dt + ∇(p - p_n) = ∇·(μ symm∇(u - u_)) - ρ (u - u_)·∇(u - u_)
        #
        # Now, if u_ is "close enough" to u, we may take the RHS to be zero (Goda, 1979;
        # see also e.g. Landet and Mortensen, 2019, section 3).
        #
        # The result is the velocity correction equation, which we will use in step 3 below:
        #
        #   ρ (u - u_) / dt + ∇p - ∇p_n = 0
        #
        # For step 2, take the divergence of the velocity correction equation, and use the continuity
        # equation to eliminate div(u) (it must be zero for the new unknown velocity); obtain a Poisson
        # problem for the new pressure p, in terms of the old pressure p_n and the tentative velocity u_:
        #
        #   -ρ ∇·u_ / dt + ∇²p - ∇²p_n = 0
        #
        # See also Langtangen and Logg (2016, section 3.4).
        #
        #   Katuhiko Goda. A multistep technique with implicit difference schemes for calculating
        #   two- or three-dimensional cavity flows. Journal of Computational Physics, 30(1):76–95,
        #   1979.
        #
        #   Tormod Landet, Mikael Mortensen, 2019. On exactly incompressible DG FEM pressure splitting
        #   schemes for the Navier-Stokes equation. arXiv: 1903.11943v1
        #
        #   Hans Petter Langtangen, Anders Logg (2016). Solving PDEs in Python: The FEniCS Tutorial 1.
        #   Simula Springer Briefs on Computing 3.
        #
        # TODO: If there are no Dirichlet BCs on p, we need to do something here to solve the
        # TODO: Poisson problem with pure-Neumann BCs. One possibility is to use a Lagrange multiplier.
        #
        # TODO: Or we could just do nothing and rely on the fact that the Krylov solvers used by FEniCS
        # TODO: can handle singular systems natively.
        # https://fenicsproject.org/olddocs/dolfin/latest/python/demos/neumann-poisson/demo_neumann-poisson.py.html
        # https://fenicsproject.org/qa/2406/solve-poisson-problem-with-neumann-bc/
        #
        # Here the LHS coefficients are constant in time.
        self.a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
        self.L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (ρ / dt) * div(u_) * q * dx

        # Define variational problem for step 3 (velocity correction)
        # Here the LHS coefficients are constant in time.
        self.a3 = ρ * dot(u, v) * dx
        self.L3 = ρ * dot(u_, v) * dx - dt * dot(nabla_grad(p_ - p_n), v) * dx

        # Assemble matrices (constant in time; do this once at the start)
        self.A1_constant = assemble(self.a1_constant)
        self.A2 = assemble(self.a2)
        self.A3 = assemble(self.a3)

        # Apply Dirichlet boundary conditions to matrices
        [bc.apply(self.A2) for bc in self.bcp]

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
        b2 = assemble(self.L2)
        [bc.apply(b2) for bc in self.bcp]
        solve(self.A2, self.p_.vector(), b2, 'bicgstab', 'hypre_amg')
        end()

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
