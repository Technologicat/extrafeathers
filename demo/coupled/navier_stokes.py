# -*- coding: utf-8; -*-
"""
Originally based on the FEniCS tutorial demo program:
Incompressible Navier-Stokes equations for flow around a cylinder
using the Incremental Pressure Correction Scheme (IPCS).

  u' + u·∇u - ∇·σ(u, p) = f
                     ∇·u = 0

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


# TODO: Add support for resuming (use TimeSeries to store u_ and p_?)

# TODO: Initialize u_ and p_ from a potential-flow approximation to have a physically reasonable initial state.
# TODO: We need to implement a potential-flow solver to be able to do that. It's just a Poisson equation,
# TODO: but the scalings must match this solver, and we need to produce a pressure field, too.
class LaminarFlow:
    """Laminar flow (incompressible Navier-Stokes) solver based on the FEniCS tutorial example.

    Uses the IPCS method (incremental pressure correction scheme; Goda 1979).

    `V`: function space for velocity
    `Q`: function space for pressure
    `ρ`: density [kg / m³]
    `μ`: dynamic viscosity [Pa s]
    `dt`: timestep [s]

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.
    """
    def __init__(self, V: VectorFunctionSpace, Q: FunctionSpace,
                 ρ: float, μ: float,
                 bcu: typing.List[DirichletBC],
                 bcp: typing.List[DirichletBC],
                 dt: float):
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

        # body force field
        self.f = Function(V)
        self.f.vector()[:] = 0.0  # placeholder value

        # initialize the underlying variables for properties
        self._dt = 1.0
        self._ρ = 1.0
        self._μ = 1.0

        # Parameters
        self.ρ = ρ
        self.μ = μ
        self.dt = dt

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

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # velocity
        u = self.u  # new (unknown)
        v = self.v  # test
        u_ = self.u_  # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # pressure
        p = self.p
        q = self.q  # test
        p_ = self.p_
        p_n = self.p_n

        # body force
        f = self.f

        # wrap constant parameters in a Constant to allow changing the value without triggering a recompile
        k = Constant(self.dt)
        μ = Constant(self.μ)
        ρ = Constant(self.ρ)

        def ε(u):  # Symmetric gradient
            return sym(nabla_grad(u))

        def σ(u, p):  # Stress tensor (isotropic Newtonian fluid)
            return 2 * μ * ε(u) - p * Identity(len(u))

        # Define variational problem for step 1 (tentative velocity)
        #
        # This is just the variational form of the momentum equation.
        #
        # The boundary terms must match the implementation of the σ term. Note we use the
        # old pressure p_n and the midpoint value of velocity U.
        #
        # Defining
        #   U = (1/2) [u_n + u]
        # we have
        #   σ(U, p_n) = 2 μ * ε(U) - p_n * I = 2 μ * ε(u_n + u) - p_n * I
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
        U = 0.5 * (u_n + u)
        F1 = (ρ * dot((u - u_n) / k, v) * dx +
              ρ * dot(dot(u_n, nabla_grad(u_n)), v) * dx +
              inner(σ(U, p_n), ε(v)) * dx +
              dot(p_n * n, v) * ds - dot(μ * nabla_grad(U) * n, v) * ds -
              dot(f, v) * dx)
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2 (pressure correction)
        #
        # Subtract the momentum equation, written in terms of tentative velocity u_ and old pressure p_n,
        # from the momentum equation written in terms of new unknown velocity u and new unknown pressure p.
        #
        # The momentum equation is
        #
        #   ρ ( ∂u/∂t + u·∇u ) = ∇·σ + f
        #                       = ∇·(μ symm∇u - p I) + f
        #                       = ∇·(μ symm∇u) - ∇p + f
        # so we have
        #
        #   ρ ( ∂(u - u_)/∂t + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Discretizing the time derivative,
        #
        #   ρ ( (u - u_n)/k - (u_ - u_n)/k + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Canceling the u_n,
        #
        #   ρ ( (u - u_)/k + (u - u_)·∇(u - u_) ) = ∇·(μ symm∇(u - u_)) - ∇(p - p_n)
        #
        # Rearranging,
        #   ρ (u - u_)/k + ∇(p - p_n) = ∇·(μ symm∇(u - u_)) - ρ (u - u_)·∇(u - u_)
        #
        # Now, if u_ is "close enough" to u, we may take the RHS to be zero (Goda, 1979;
        # see also e.g. Landet and Mortensen, 2019, section 3).
        #
        # The result is the velocity correction equation, which we will use in step 3 below:
        #
        #   ρ (u - u_) / k + ∇p - ∇p_n = 0
        #
        # For step 2, take the divergence of the velocity correction equation, and use the continuity
        # equation to eliminate div(u) (it must be zero for the new unknown velocity); obtain a Poisson
        # problem for the new pressure p, in terms of the old pressure p_n and the tentative velocity u_:
        #
        #   -ρ ∇·u_ / k + ∇²p - ∇²p_n = 0
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
        self.a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
        self.L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (ρ / k) * div(u_) * q * dx

        # Define variational problem for step 3 (velocity correction)
        self.a3 = ρ * dot(u, v) * dx
        self.L3 = ρ * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

        # Assemble matrices (constant in time; do this once at the start)
        self.A1 = assemble(self.a1)
        self.A2 = assemble(self.a2)
        self.A3 = assemble(self.a3)

        # Apply Dirichlet boundary conditions to matrices
        [bc.apply(self.A1) for bc in self.bcu]
        [bc.apply(self.A2) for bc in self.bcp]

    def step(self) -> None:
        """Take a timestep of length `self.dt`.

        Updates `self.u_` and `self.p_`.
        """
        # Step 1: Tentative velocity step
        begin("Tentative velocity")
        b1 = assemble(self.L1)
        [bc.apply(b1) for bc in self.bcu]
        solve(self.A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg')
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
