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
                    dot, inner,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve, project,
                    interpolate, normalize, VectorSpaceBasis, as_backend_type,
                    begin, end)

from ..meshfunction import meshsize, cell_mf_to_expression
from .numutil import ε, advw, advs, mag
from .util import ufl_constant_property, StabilizerFlags


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
            '''Deviatoric (traceless) part of rank-2 tensor `T`.

            This assumes, for 2D, that `T` is actually 3D, but
            the third row and column of `T` are zero.
            '''
            return T - (1 / 3) * tr(T) * Identity(T.geometric_dimension())
        # `solver._μ` is the UFL `Constant` object
        σ = σ(solver.u_, solver.p_, solver._μ)
        s = dev(σ)
        vonMises = sqrt(3 / 2 * inner(s, s))
        plot(project(vonMises, W))
    """
    return 2 * μ * ε(u) - p * Identity(p.geometric_dimension())


class NavierStokesStabilizerFlags(StabilizerFlags):
    """Interface for numerical stabilizer on/off flags.

    Collects them into one namespace; handles translation between
    `bool` values and the UFL expressions that are actually used
    in the equations.

    Usage::

        print(solver.stabilizers)  # status --> "<NavierStokesStabilizerFlags: LSIC(True), SUPG(True)>"
        solver.stabilizers.SUPG = True  # enable SUPG
        solver.stabilizers.SUPG = False  # disable SUPG
    """
    def __init__(self):  # set up the UFL expressions for the flags
        super().__init__()
        self._LSIC = Expression('b', degree=0, b=1.0)
        self._SUPG = Expression('b', degree=0, b=1.0)

    def _get_LSIC(self) -> bool:
        return bool(self._LSIC.b)
    def _set_LSIC(self, b: bool) -> None:
        self._LSIC.b = float(b)
    LSIC = property(fget=_get_LSIC, fset=_set_LSIC, doc="Least-squares incompressibility, for additional stability at high Re.")

    def _get_SUPG(self) -> bool:
        return bool(self._SUPG.b)
    def _set_SUPG(self, b: bool) -> None:
        self._SUPG.b = float(b)
    SUPG = property(fget=_get_SUPG, fset=_set_SUPG, doc="Streamline upwinding Petrov-Galerkin, for advection-dominant flows.")


# TODO: use nondimensional form
class NavierStokes:
    """Incompressible Navier-Stokes solver, no turbulence model.

    Main use case is laminar flow. Can be used for direct numerical simulation (DNS)
    of turbulent flows, if the mesh is fine enough.

    Uses the IPCS method (incremental pressure correction scheme; Goda, 1979),
    as presented in the FEniCS tutorial. Stabilized with SUPG, LSIC, and
    skew-symmetric advection. See Donea & Huerta (2003). Can handle cases
    with pure Neumann BCs on pressure.

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

         Note that for θ = 0, the SUPG stabilization parameter τ_SUPG → 0,
         so when using forward Euler, it does not make sense to enable the
         SUPG stabilizer.

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.

    The specific body force `self.f` [N / kg] = [m / s²] is an assignable FEM function.
    Interpolate or project what you want onto `V`, and then `.assign`. For example,
    to set a constant body force everywhere::

        f: Function = interpolate(Constant((0, -9.81)), V)
        solver.f.assign(f)

    (Note that if you do simulate gravity, you may need to modify some of your boundary
    conditions to accommodate. Specifically, any Dirichlet boundary conditions on pressure
    must account for the hydrostatic pressure, p = ρ g h.)

    **Allowed element types**:

    Because incompressible flow is a saddle-point problem, the function spaces
    `V` and `Q` must be LBB-compatible, i.e. they must satisfy the inf-sup condition
    (a.k.a. the Ladyzhenskaya–Babuška–Brezzi condition) for the linear form
    b(v, q) := ∫ (∇·v) q dΩ (see Brenner & Scott, sec. 12.5, 12.6).

    Taylor-Hood (a.k.a. P2P1) elements are the classical LBB-compatible choice;
    choose `P2` for `V` and `P1` for `Q`. Q2Q1 works, too.

    It is nontrivial to tell whether an arbitrary pair of `V` and `Q` are LBB-compatible
    or not. A typical symptom of an LBB violation are high spatial frequency numerical
    pressure oscillations.

    Non-compatible discretizations (e.g. P1P1 or Q1Q1) can be made to work by postprocessing
    the pressure field between timesteps with a least-squares smoother, thereby reducing
    oscillations at the scale of the local element size. This technique is sometimes
    used in elasticity, where mixed formulations run into the same issue. See Hughes,
    sec. 4.4.1 and appendix 4.II. Hughes particularly notes that in order for the smoothing
    to have the desired effect, it should be based on a least-squares approach.

    This solver automatically least-squares-smooths the pressure if `V` is a P1 space.
    If you need to perform such smoothing manually for some reason, you can e.g.::

        from dolfin import project, interpolate, FunctionSpace
        Qproj = FunctionSpace(Q.mesh(), "DG", 0)

    where `Q` is your pressure function space, and then in your timestep loop::

        solver.step()
        solver.p_.assign(project(interpolate(solver.p_, Qproj), Q))
        solver.commit()

    where `solver` is your `NavierStokes` instance. This is how the solver does it.

    Technically, to be fully correct::

        project(project(solver.p_, Qproj), Q)

    but when `Q` is a P1 space, `project(interpolate(...))` produces the same result,
    while being faster, since it only needs to solve one linear equation system (not two).

    Classical patch averaging (projecting or interpolating onto dG0 and then averaging
    the resulting piecewise constant field) is also available, as `patch_average`.


    **References**:

        Susanne C. Brenner & L. Ridgway Scott. 2010. The Mathematical Theory
        of Finite Element Methods. 3rd edition. Springer. ISBN 978-1-4419-2611-1.

        Jean Donea and Antonio Huerta. 2003. Finite Element Methods for Flow Problems.
        Wiley. ISBN 0-471-49666-9.

        Katuhiko Goda. A multistep technique with implicit difference schemes for
        calculating two- or three-dimensional cavity flows. Journal of Computational
        Physics, 30(1):76–95, 1979.

        Thomas J. R. Hughes. 2000. The Finite Element Method: Linear Static and Dynamic
        Finite Element Analysis. Dover. Corrected and updated reprint of the 1987 edition.
        ISBN 978-0-486-41181-1.
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
        self.dG0 = FunctionSpace(self.mesh, "DG", 0)  # for P1P1 pressure postprocessing
        self.bcu = bcu
        self.bcp = bcp

        # Function space of ℝ (single global DOF), for computing the average pressure.
        self.W = FunctionSpace(self.mesh, "R", 0)

        # Set up the null space of pressure correction subproblem.
        # We'll need this in cases with only Neumann BCs on pressure.
        #
        # https://fenicsproject.org/qa/4121/help-parallelising-space-method-eliminating-rigid-motion/
        # https://www.mail-archive.com/fenics@fenicsproject.org/msg00912.html
        # https://www.mail-archive.com/fenics@fenicsproject.org/msg00909.html
        # https://fenicsproject.org/olddocs/dolfin/latest/python/demos/singular-poisson/demo_singular-poisson.py.html
        ns = [Constant(1)]  # The null space are the constant functions.
        null_space_basis = [interpolate(nk, Q).vector() for nk in ns]
        [normalize(nk_vec, 'l2') for nk_vec in null_space_basis]
        self.pressure_null_space = VectorSpaceBasis(null_space_basis)

        # Trial and test functions
        self.u = TrialFunction(V)  # no suffix: the UFL symbol for the unknown quantity
        self.v = TestFunction(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)

        # Functions for solution at previous and current time steps
        self.u_n = Function(V, name="u_n")  # suffix _n: the old value (end of previous timestep)
        self.u_ = Function(V, name="u")  # suffix _: the latest computed approximation
        self.p_n = Function(Q, name="p_n")
        self.p_ = Function(Q, name="p")

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Specific body force. FEM function for maximum generality.
        # Note this is the value at the θ-point inside the timestep.
        self.f = Function(V, name="f")
        self.f.vector()[:] = 0.0  # placeholder value

        # Parameters.
        self._ρ = Constant(ρ)
        self._μ = Constant(μ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = NavierStokesStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    μ = ufl_constant_property("μ", doc="Dynamic viscosity [Pa s]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def reynolds(self, u: float, L: float) -> float:
        """Return the Reynolds number of the flow.

        `u`: characteristic speed (scalar) [m / s]
        `L`: length scale [m]

        The Reynolds number is defined as the ratio of advective vs.
        diffusive effects::

            Re = u L / ν

        where `ν = μ / ρ` is the kinematic viscosity.

        Choosing representative values for `u` and `L` is more of an art
        than a science. Typically:

            - In a flow past an obstacle, `u` is the free-stream speed,
              and `L` is the length of the obstacle along the direction
              of the free stream.

            - In a lid-driven cavity flow in a square cavity, `u` is the
              speed of the lid, and `L` is the length of the lid.

            - In a flow in a pipe, `u` is the free-stream speed, and `L`
              is the pipe diameter.

            - For giving the user a dynamic estimate of `Re` during computation,
              the maximum value of `|u|` may be generally useful, but `L`
              must still be intuited from the problem geometry.
        """
        ν = self.μ / self.ρ  # kinematic viscosity,  [ν] = m² / s
        return u * L / ν

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

        enable_LSIC_flag = self.stabilizers._LSIC
        enable_SUPG_flag = self.stabilizers._SUPG

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
        #   σ(U) : ∇v dx = σ(U) : symm∇(v) dx = σ(U) : ε(v) dx
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
        F1_varying = (ρ * (1 / 2) * advw(a, ustar, v, n, mode="divergence-free"))
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
        # On the other hand, by comparing to isotropic linear elasticity, where
        # the stress term can be written as (see e.g. Brenner & Scott, 2010, p. 314)
        #   ∫ 2 μ ε(u) : ε(v) + λ div(u) div(v) dΩ
        # where μ and λ are the Lamé parameters; we can conclude that for the flow
        # problem, the  div(u) div(v)  term represents a volumetric viscosity. Thus,
        # it tends to resist compression and expansion. (Keep in mind that in the
        # case of a flow, `u` is a velocity, not a displacement, so this will
        # actually resist *accelerations* of this kind.)
        #
        # Reference:
        #   Susanne C. Brenner & L. Ridgway Scott. 2010. The Mathematical Theory
        #   of Finite Element Methods. 3rd edition. Springer. ISBN 978-1-4419-2611-1.
        #
        # Note [grad(div(u))] = (1 / m²) (m / s) = 1 / (m s)
        # so [τ_LSIC] [grad(div(u))] = (m² / s) (1 / (m s)) = m / s²,
        # an acceleration; thus matching the other terms of the strong form.
        #
        # Note we use the unknown `u`, not the `U` used in the θ method; only
        # the solution at the end of the timestep should be penalized for
        # deviation from known properties of the true solution.
        τ_LSIC = mag(a) * he / 2  # [τ_LSIC] = (m / s) * m = m² / s, a kinematic viscosity
        F1_LSIC = enable_LSIC_flag * τ_LSIC * div(u) * div(v) * dx
        F1_varying += F1_LSIC

        # SUPG: streamline upwinding Petrov-Galerkin.
        #
        # Residual-based, consistent. Stabilizes the Galerkin formulation in the presence
        # of a dominating convective term in the momentum equation.
        #
        # See Donea & Huerta (2003), sec. 6.7.2 (p. 296), 6.5.8 (p. 287),
        # and 5.4.6.2 (p. 232). See also sec. 2.4, p. 59 ff.

        # [μ] / [ρ] = (Pa s) / (kg / m³)
        #           = (N s / m²) / (kg / m³)
        #           = (kg m s / (m² s²)) / (kg / m³)
        #           = (kg / (m s)) * m³ / kg
        #           = m² / s,  kinematic viscosity
        # Donea & Huerta (2003, p. 232), based on Shakib et al. (1991).
        # Donea & Huerta, p. 288: "α₀ = 1 / 3 appears to be optimal for linear elements"
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (μ / ρ) / he**2)**-1  # [τ] = s
        def R(U_, P_):
            # The residual is evaluated elementwise in strong form.
            return ρ * ((U_ - u_n) / dt + advs(a, U_, mode="divergence-free")) - div(σ(U_, P_, μ)) - ρ * f
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, v, mode="divergence-free"), R(u, p_n)) * dx
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
        # natively (as long as we tell them about the nullspace). We will
        # postprocess to zero out the average pressure (to make the pressure
        # always unique).
        #
        # https://fenicsproject.org/qa/2406/solve-poisson-problem-with-neumann-bc/
        #
        # Using a Lagrange multiplier is another option. Mathematically elegant,
        # but numerically evil, as it introduces a full row/column, and destroys symmetry.
        # (To be fair, we don't have symmetry anyway due to PSPG stabilization;
        # but we can avoid the full row/column.)
        #
        # https://fenicsproject.org/olddocs/dolfin/latest/python/demos/neumann-poisson/demo_neumann-poisson.py.html
        #
        # (To see where the weak forms for the Lagrange multiplier term come
        # from, consider the Lagrangian associated with the PDE; add the term
        # λ * <constraint>; multiply by test function; integrate over Ω; take the
        # Fréchet derivatives (or alternatively, Gateaux derivatives) w.r.t.
        # `u` and `λ` and sum them to get the Euler-Lagrange equation; group
        # terms by test function. Terms with `q` and `∇q` go into the same equation,
        # because although linearly independent, they are dependent in another sense:
        # choosing the arbitrary function `q` fully determines `∇q`.)
        #
        # Here the LHS coefficients are constant in time.
        self.a2_constant = dot(nabla_grad(p), nabla_grad(q)) * dx
        self.L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx(self.Q.mesh()) - (ρ / dt) * div(u_) * q * dx

        # PSPG: pressure-stabilizing Petrov-Galerkin.
        #
        # Used together with SUPG, for some schemes this allows the use of
        # LBB-incompatible elements; but unfortunately not for IPCS (Donea and
        # Huerta, 2003, remark 6.22, pp. 303-304).
        #
        # We will still apply PSPG in hopes that it helps stability at high Re;
        # at least we then treat both equations consistently.
        #
        # Consistent, residual-based. Note the method uses the residual of the
        # *momentum* equation, but now with the tentative velocity `u_` and the
        # unknown pressure `p`.
        #
        # There was no value for τ_PSPG given in the book, so we recycle τ_SUPG here.
        # There was a reference, though, to a paper by Tezduyar & Osawa (2000),
        # which discusses element-local stabilization parameter values in more detail;
        # I'll need to check that out later (see p. 297 in the book for the reference).
        #
        # (Maybe the exact value doesn't matter that much; these stabilization methods
        #  usually work as long as the `he` and `Δt` scalings are correct.)
        τ_PSPG, enable_PSPG_flag = τ_SUPG, enable_SUPG_flag
        F_PSPG = enable_PSPG_flag * τ_PSPG * dot(nabla_grad(q), R(u_, p)) * dx
        self.a2_varying = lhs(F_PSPG)
        self.L2 += rhs(F_PSPG)

        # Define variational problem for step 3 (velocity correction)
        # Here the LHS coefficients are constant in time.
        self.a3 = ρ * dot(u, v) * dx
        self.L3 = ρ * dot(u_, v) * dx - dt * dot(nabla_grad(p_ - p_n), v) * dx(self.V.mesh())

        # Assemble matrices (constant in time; do this once at the start)
        self.A1_constant = assemble(self.a1_constant)
        self.A2_constant = assemble(self.a2_constant)
        self.A3 = assemble(self.a3)

    def step(self) -> typing.Tuple[int, int, int]:
        """Take a timestep of length `self.dt`.

        Updates `self.u_` and `self.p_`.

        Returns `(it1, it2, it3)`, where `itK` is the number of
        Krylov iterations taken by step `K` of the IPCS algorithm:

          - Step 1: momentum equation (tentative velocity)
                    (non-symmetric, bicgstab with AMG preconditioner)
          - Step 2: incremental pressure correction
                    (non-symmetric, bicgstab with AMG preconditioner)
          - Step 3: incremental velocity correction
                    (symmetric, cg with SOR preconditioner)

        Note that due to different algebraic structure and different
        Krylov algorithms, the numbers from the three IPCS steps are
        not directly comparable.
        """
        # Step 1: Tentative velocity step
        begin("Tentative velocity")
        # For P3 velocity in 2D, FEniCS may use over 100 quadrature points per cell
        # when assembling `A1` and `b1`, so that the integrals are exact for the
        # highest-degree polynomials in the form. We could tune this using
        # `form_compiler_parameters`, for example:
        #     A = assemble(form, form_compiler_parameters={"quadrature_degree": 2})
        # https://fenicsproject.org/qa/10991/expression-degree-and-quadrature-rules/
        # https://fenicsproject.org/qa/7415/question-concerning-the-number-of-integration-points/
        A1 = self.A1_constant + assemble(self.a1_varying)
        b1 = assemble(self.L1)
        [bc.apply(A1) for bc in self.bcu]
        [bc.apply(b1) for bc in self.bcu]
        it1 = solve(A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg')
        end()

        # Step 2: Pressure correction step
        begin("Pressure correction")
        A2 = self.A2_constant + assemble(self.a2_varying)
        b2 = assemble(self.L2)
        if not self.bcp:  # pure Neumann pressure BCs
            # `set_near_nullspace`: "Attach near nullspace to matrix (used by preconditioners,
            #                        such as smoothed aggregation algebraic multigrid)"
            # `set_nullspace`:      "Attach nullspace to matrix (typically used by Krylov solvers
            #                        when solving singular systems)"
            as_backend_type(A2).set_near_nullspace(self.pressure_null_space)
            as_backend_type(A2).set_nullspace(self.pressure_null_space)
            self.pressure_null_space.orthogonalize(b2)
        else:
            [bc.apply(A2) for bc in self.bcp]
            [bc.apply(b2) for bc in self.bcp]
        it2 = solve(A2, self.p_.vector(), b2, 'bicgstab', 'hypre_amg')
        end()

        if not self.bcp:
            # Step 2½: In cases with pure Neumann BCs on pressure (e.g.
            # standard cavity flow test cases), zero out the average.
            # This defines the pressure uniquely.
            #
            # If there is at least one Dirichlet BC on pressure, we skip this
            # step. One Dirichlet BC is sufficient to make the pressure unique,
            # and it is better (for least surprise, from a usability perspective)
            # to then actually produce a solution that satisfies the given Dirichlet BCs.
            #
            # Strategy:
            #
            # Shifting `p` by a constant does not matter as far as the momentum
            # equation is concerned. We L2-project `p_` onto ℝ, and then subtract
            # the result from the original `p_`. Thus the mean of the shifted `p_`
            # will be zero.
            #
            # How to extract the single value of a `Function` on Reals (single
            # global DOF) is not documented very well. See the source code of
            # `dolfin.Function`, it has a `__float__` method. The sanity checks
            # indicate it is intended precisely for this.
            begin("Zero out mean pressure")
            avgp = project(self.p_, self.W)
            self.p_.vector()[:] -= float(avgp)
            end()

        # Step 3: Velocity correction step
        begin("Velocity correction")
        b3 = assemble(self.L3)
        it3 = solve(self.A3, self.u_.vector(), b3, 'cg', 'sor')
        end()

        # EXPERIMENTAL:
        # If P1P1 discretization (which does not satisfy the LBB condition),
        # postprocess the pressure to kill off the checkerboard mode.
        #
        # Doing this after step 3 yields a velocity field with fewer spurious oscillations
        # (especially at the start of the simulation, when the field does not yet satisfy
        # the PDE) than if we did this between steps 2 and 2½.
        if self.V.ufl_element().degree() == 1:
            begin("Smooth pressure")
            if self.Q.ufl_element().degree() == 1:
                # When pressure is P1, sampling at the element midpoint (which is also the dG0
                # integration point) is a cheap way to get the average of the field over the element.
                # This saves us one linear solve.
                self.p_.assign(project(interpolate(self.p_, self.dG0), self.Q))
            else:
                # Otherwise, we must project to dG0 to obtain the average.
                self.p_.assign(project(project(self.p_, self.dG0), self.Q))
            # # alternative: classical patch averaging (see docstring for setting up the extra args)
            # self.p_.assign(meshmagic.patch_average(self.p_, self.dG0, QtodG0, dG0tocell, cell_volume))
            end()

        return it1, it2, it3

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This copies `self.u_` to `self.u_n` and `self.p_` to `self.p_n`,
        making the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.
        """
        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)
