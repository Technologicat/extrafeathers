# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of uniform axial motion).

Kelvin--Voigt constitutive model.

This advanced version includes the thermal expansion effect, adding the temperature field as a driver.
"""

__all__ = ["LinearMomentumBalance",
           "InternalEnergyBalance",
           "step_adaptive"]

import typing

from fenics import (VectorFunctionSpace, TensorFunctionSpace,
                    MixedElement, FunctionSpace, TrialFunctions, TestFunctions, split,
                    TrialFunction, TestFunction,
                    Constant, Expression, Function,
                    project, FunctionAssigner,
                    FacetNormal, DirichletBC,
                    Measure, MeshFunction,
                    dot, inner, sym, tr,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve,
                    begin, end)

from ..meshfunction import meshsize, cell_mf_to_expression
from .numutil import Maxx, ε, mag, advw, advs
from .util import ufl_constant_property, StabilizerFlags as StabilizerFlagsBase

from .eulerian_solid import step_adaptive  # re-export


class StabilizerFlags(StabilizerFlagsBase):
    """Interface for numerical stabilizer on/off flags.

    Collects them into one namespace; handles translation between
    `bool` values and the UFL expressions that are actually used
    in the equations.

    Usage::

        print(solver.stabilizers)  # status --> "<EulerianSolidAdvancedStabilizerFlags: SUPG(True)>"
        solver.stabilizers.SUPG = True  # enable SUPG
        solver.stabilizers.SUPG = False  # disable SUPG
    """
    def __init__(self):  # set up the UFL expressions for the flags
        super().__init__()
        self._SUPG = Expression('b', degree=0, b=1.0)

    def _get_SUPG(self) -> bool:
        return bool(self._SUPG.b)
    def _set_SUPG(self, b: bool) -> None:
        self._SUPG.b = float(b)
    SUPG = property(fget=_get_SUPG, fset=_set_SUPG, doc="Streamline upwinding Petrov-Galerkin, for advection-dominant problems.")


class LinearMomentumBalance:
    """Eulerian linear momentum balance law for an axially moving Kelvin-Voigt material.

    (A.k.a. Cauchy's first law, specialized to a Kelvin-Voigt material.)

    This model includes the thermal expansion effect.

    Like `extrafeathers.pdes.eulerian_solid.EulerianSolidPrimal`, but with thermal expansion.
    If you don't need the thermal features, prefer `EulerianSolidPrimal`; it runs faster.

    The linear momentum balance law is used for determining the displacement `u` [m].
    We use a mixed-Lagrangean-Eulerian (MLE) description. The equation itself is written
    in the laboratory frame `x`, but the displacement field `u(x, t)` is measured against
    a reference state that translates at a constant axial velocity.

    The velocity variable `v` [m/s] is determined by L2 projection, using the definition:

      v := du/dt = ∂u/∂t + (a·∇)u

    where `a` is the constant axial velocity field. That is, `v` is the velocity of the
    material parcels with respect to the *axially co-moving frame*.

    The projection is performed using the *unknown* `u`, by including this equation
    into a monolithic equation system.

    Boundary conditions should be set on `u`; `v` takes no BCs.

    Boundary stresses are enforced using a Neumann BC. `bcσ` is a single expression
    that will be evaluated at boundaries that do not have a boundary condition for `u`.


    Thermal features:

    Thermal behavior is typically nonlinear, because the material properties that affect it are
    temperature-dependent. To solve the coupled thermomechanical problem, we use a weakly coupled
    (i.e. system iteration) approach.

    To this end --- although the mechanical subproblem itself is linear --- this solver supports
    an iterative Picard scheme. Calling `.step()` computes the next Picard iterate. It can be called
    several times to iteratively refine the solution. Call `.commit()` only after you are satisfied
    with the result for the current timestep. The previous Picard iterate is available in `self.s_prev`,
    for convergence monitoring by user.

    Note the Picard iteration feature only exists so that we can iteratively refine the effects of
    thermomechanical coupling; since the mechanical subproblem is linear, its solution (for any given
    thermal fields) is always reached in a single step.

    The thermal subproblem is nonlinear, but keep in mind that after even one step of the thermal
    subproblem, the stress and velocity fields will be out of sync with the updated thermal fields.
    Thus rather than iterate the thermal subproblem to convergence, it may be more efficient to
    step the thermal problem just once at each system iteration, to maximize communication and
    consistency between the subproblems --- and iterate only the complete system to convergence.

    At each system iteration, you'll need to feed this solver with `T` and `dT/dt` from the thermal
    solver, and the thermal solver with `a + v` (full velocity field measured against the laboratory
    frame) and `σ` (Cauchy stress) from this solver. Note the coupling is bidirectional; each solver
    affects the other one.

    Explicitly: step this solver, then call `.export_stress()`, and send in the computed velocity
    and stress to the thermal solver. Step the thermal solver, and then send the updated thermal
    fields here. This completes one system iteration.

    Perform more system iterations until the solution converges, and commit the timestep (for both
    solvers) only after it has converged.

    The beginning-of-timestep value for the temperature field is read from `self.T_n`, and the
    end-of-timestep value is read from `self.T_`. Within the timestep (for the θ time integrator),
    these values are interpolated linearly.

    Before running the first timestep, it is important to initialize both the old and current
    thermal fields. When the solver is instantiated, the default initial value for both of these
    temperature fields is the constant `T0`, and for corresponding the `dT/dt` fields, the default
    is the constant zero.

    During later timesteps, it is sufficient to send the new thermal fields. When you `.commit()`
    a timestep, the latest fields are automatically copied to the old fields, and the "old old"
    fields are discarded.


    Parameters:

        `V`: vector function space for `u` and `v`
        `Q`: tensor function space for visualizing `σ` (visualization/export only)
        `P`: tensor function space for visualizing `ε` (visualization/export only)
        `ρ`: density [kg/m³]
        `λ`: Lamé's first parameter [Pa]; temperature-dependent, callable: T -> Expression
        `μ`: shear modulus [Pa]; ; temperature-dependent, callable: T -> Expression
        `τ`: Kelvin-Voigt retardation time [s]
             float; if zero, viscous terms are omitted when the solver initializes,
             producing an elastothermal model.
        `α`: thermal expansion tensor [1/K]
             Symmetric, rank 2. For isotropic material, use  α' * Identity(2),
             where α' is the scalar coefficient of linear thermal expansion.
        `dαdT`: ∂α/∂T [1/K²]
        `T0`: reference temperature for thermal expansion, at which
              the thermal expansion effect is considered zero [K]

              The value is stored as a UFL `Constant` in `self._T0`; this can be
              used in the expressions for the temperature-dependent parameters.

        `V0`: axial drive velocity in the +x direction [m/s]
        `bcu`: Dirichlet boundary conditions for `u` [m]
        `bcσ`: Neumann boundary condition for `σ`. [Pa]

               The Neumann boundary is defined as that part of the domain boundary
               for which no Dirichlet BCs for `u` have been given.

               Any part of the Neumann boundary for which a Neumann BC expression
               is not given, defaults to zero Neumann (i.e. free of tractions).

               The format is `[(fenics_expression, boundary_tag or None), ...]` where
               `None` means to use the expression on the whole Neumann boundary.

               A `boundary_tag` is an `int` that matches a boundary number in
               `boundary_parts`, which see. This means to use the expression
               on the specified part of the Neumann boundary only.

               IMPORTANT: the Neumann BC expressions are compiled just once, when the solver
               is instantiated. If you need to update a coefficient inside a Neumann BC
               during a simulation, give your `Expression` some updatable parameters::

                   axial_pull_strength = lambda t: 1e8 * min(t, 1.0)  # for example
                   stress_neumann_bc = Expression((("σ11", "0"), ("0", "0")),
                                                  degree=1,
                                                  σ11=axial_pull_strength(0.0))

                   # ...much later, during timestep loop...

                   stress_neumann_bc.σ11 = axial_pull_strength(t)

               And if you prefer to populate the Neumann BC list after instantiation
               (passing in an empty list initially, then adding BCs to it later),
               call `.compile_forms()` to refresh the equations to use the latest
               definitions.

        `dt`: timestep size [s]
        `θ`: parameter of the theta time integrator.

        `boundary_parts`: A facet `MeshFunction` that numbers the physical boundaries in the problem
                          (such as left edge, inlet, etc.).

                          This is the same `boundary_parts` that is used (in the main program of
                          a solver) to specify which boundaries Dirichlet BCs apply to; see the
                          example solvers. Produced by mesh generation with a suitable setup.

        `temperature_degree`: The degree of the finite element space of the temperature field.
                              The default `None` means "use the same degree as `V`".

    External field inputs:
        All these fields live on a scalar element of the same kind as `V`,
        of degree `temperature_degree`.

        The degree must match the data you are sending in!

        Use `solver.xxx.assign(...)` to send in field values, where
        `xxx` is `T_`, `dTdt_`, `T_n`, or `dTdt_n`, as appropriate.

        `T_`: temperature field [K]

        `dTdt_`: *material* derivative of temperature [K/s].

                 This is an Eulerian solver. If you are looking at an axially moving
                 continuum, please account for the axial motion. If you use the
                 `InternalEnergyBalance` solver to compute `T` and `dT/dt`,
                 it does so automatically.

                 How it works: the material derivative of the temperature is
                     dT/dt = ∂T/∂t + ((a + v)·∇)T
                 where `a` is the (constant) axial drive velocity field, and `v` is
                 the velocity field of the material parcels as measured against the
                 axially co-moving frame. Therefore, `a + v` is the velocity field
                 of the material parcels as measured against the laboratory frame.

                 When the field T is known, its material derivative dT/dt
                 can be computed by solving the PDE
                     dT/dt - ∂T/∂t - ((a + v)·∇)T = 0
                 This admits an L2 projection using FEM (no BCs needed):
                     (dT/dt, w) - (∂T/∂t, w) - (((a + v)·∇)T, w) = 0
                      unknown        known            known
                 where w is the test function corresponding to the unknown field dT/dt.

                 The directional derivative term can be numerically stabilized
                 using skew-symmetric advection and SUPG. The `InternalEnergyBalance`
                 solver uses these techniques.

        `T_n`, `dTdt_n`: old fields at beginning of timestep (at the first
                         timestep, the initial fields)
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: typing.Callable, μ: typing.Callable, τ: float,
                 α: typing.Callable, dαdT: typing.Callable, T0: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcσ: typing.List[typing.Tuple[Expression, typing.Optional[int]]],
                 dt: float, θ: float = 0.5, *,
                 boundary_parts: typing.Optional[MeshFunction] = None,
                 temperature_degree: int = None):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Set up subdomains for the boundary parts.
        #
        # In the Neumann BC, to use different expressions on different boundaries, we must set up subdomains for `ds`.
        # We must split `ds` to apply BCs selectively by boundary tag, and include a list of boundary tags in the
        # Neumann BC specification.
        #
        # See the tutorial:
        #   https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html#fenics-implementation-14
        # particularly, how to redefine the measure `ds` in terms of boundary markers.
        #
        # Then we can use `ds(i)` as the integration symbol, where `i` is the boundary number
        # in the mesh data, e.g. `Boundaries.LEFT.value` from the problem configuration.
        # Using bare `ds`, as usual, applies the BC to the whole Neumann boundary.
        #
        self.boundary_parts = boundary_parts
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_parts) if boundary_parts is not None else None

        e = MixedElement(V.ufl_element(), V.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        w, ψ = TestFunctions(S)
        s_ = Function(S)  # suffix _: latest computed approximation
        u_, v_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, v_ = s_.sub(0), s_.sub(1)  # if you want the `Function` (for plotting etc.)
        s_n = Function(S)  # suffix _n: old value (end of previous timestep)
        u_n, v_n = split(s_n)

        # Previous Picard iterate, for convergence monitoring by user
        s_prev = Function(S)

        self.u, self.v = u, v  # trials
        self.w, self.ψ = w, ψ  # tests
        self.u_, self.v_ = u_, v_  # latest computed approximation
        self.u_n, self.v_n = u_n, v_n  # old value (end of previous timestep)

        # Whole mixed vector function space
        self.S = S
        self.s_ = s_
        self.s_n = s_n
        self.s_prev = s_prev

        # Original vector function space, used for building the mixed vector function space
        self.V = V

        # We need a scalar function space for the temperature field.
        T_degree = temperature_degree if temperature_degree is not None else V.ufl_element().degree()
        V_rank0 = FunctionSpace(V.mesh(), V.ufl_element().family(), T_degree)
        self.V_rank0 = V_rank0

        # Material parameters.
        # Single-argument functions: T -> Expression.
        #
        # As for the thermal properties:
        #
        # - The reference temperature constant is available as `self._T0`,
        #   and can be used in the expression.
        # - The default temperature is T ≡ T0 everywhere and at all times,
        #   so that no thermal effects appear unless these are set explicitly.
        #
        # TODO: Parameterize the (rank-4) stiffness/viscosity tensor `K`,
        #       not isotropic Lamé parameters (better for arbitrary symmetry group).
        self.λ = λ
        self.μ = μ
        self.α = α  # thermal expansion tensor (rank-2, symmetric)
        self.dαdT = dαdT  # ∂α/∂T (rank-2, symmetric), for part of viscothermal response

        # Temperature field (driver).
        #
        # For consistency, in the θ time integrator, we evaluate the thermal fields
        # at the same point of time as the mechanical fields. Thus we need both
        # old and new values.
        T_ = Function(V_rank0)   # new value (end of timestep)
        T_n = Function(V_rank0)  # old value (start of timestep)
        T_.vector()[:] = T0
        T_n.vector()[:] = T0
        self.T_ = T_
        self.T_n = T_n

        # *Material* derivative of temperature, for viscothermal response.
        # This really is `dT/dt`, NOT `∂T/∂t`. And note this is a *material*
        # derivative, not just a co-moving derivative; the velocity field
        # is not just the axial drive velocity field `a`, but `a + v`, where `v`
        # is the velocity of the material parcels in the co-moving frame.
        #
        # That is, the velocity that should be used for computing dT/dt is the
        # velocity of the material parcels in the *laboratory* frame.
        #
        # When solving a thermomechanical problem, this can come from an auxiliary variable
        # for the internal energy balance law, playing a role similar to our `v`.
        # You can obtain this from the `InternalEnergyBalance` solver.
        dTdt_ = Function(V_rank0)
        dTdt_n = Function(V_rank0)  # old value
        dTdt_.vector()[:] = 0.0
        dTdt_n.vector()[:] = 0.0
        self.dTdt_ = dTdt_
        self.dTdt_n = dTdt_n

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        # θ integration, so we need both old and new values for this too.
        self.b_ = Function(V)
        self.b_.vector()[:] = 0.0
        self.b_n = Function(V)
        self.b_n.vector()[:] = 0.0

        # Dirichlet boundary conditions
        self.bcu = bcu

        # Neumann BC for stress
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.V0 = V0
        self.a = Constant((V0, 0))

        # Constant parameters.
        self._ρ = Constant(ρ)
        self._τ = Constant(τ)
        self._T0 = Constant(T0)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # For separate equations, for stress, strain and strain rate visualization/export.
        # We use the trial and test functions to compute the L2 projection for visualization.
        # This algorithm uses the spaces `Q` and `P` only for visualization/export.
        self.Q = Q
        self.σ = TrialFunction(Q)
        self.φ = TestFunction(Q)
        self.σ_ = Function(Q)
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        # Numerical stabilizer on/off flags.
        self.stabilizers = StabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
    T0 = ufl_constant_property("T0", doc="Reference temperature for thermal expansion [K]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u = self.u      # new (unknown)
        w = self.w      # test
        u_ = self.u_    # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Material parcel velocity in co-moving frame,  v = ∂u/∂t + (a·∇)u
        # where `a` is the axial drive velocity (the velocity of the co-moving
        # frame as measured against the laboratory frame).
        v = self.v
        ψ = self.ψ
        v_ = self.v_
        v_n = self.v_n

        # Temperature (driver)
        T_ = self.T_
        T_n = self.T_n

        # *Material* derivative of temperature (driver).
        # Note that the consistent first-order approximation is
        #   dT/dt = ∂T/∂t + ((a + v)·∇)T
        # so here the velocity field is not just `a`, but `a + v`.
        dTdt_ = self.dTdt_
        dTdt_n = self.dTdt_n

        # Specific body force
        b_ = self.b_
        b_n = self.b_n

        # Stress (visualization/export only)
        σ = self.σ  # unknown
        φ = self.φ  # test (in space `Q`)

        # Strain and strain rate (visualization/export only)
        εu = self.εu  # strain, unknown
        εv = self.εv  # strain rate (at material parcel), unknown
        # These are solved separately in the same function space, so we can re-use the same test `q`.
        q = self.q  # test (in space `P`)

        # Velocity field for axial motion
        a = self.a

        # Local mesh size (for stabilization terms)
        he = self.he

        # Constant parameters
        ρ = self._ρ
        τ = self._τ
        T0 = self._T0
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # The strong form of the equations we are discretizing is
        # (constant-temperature variant shown here for simplicity):
        #
        #   V = ∂u/∂t + (a·∇) u   [velocity variable]
        #
        #   σ = E : ε + η : dε/dt
        #     = E : (symm ∇u) + η : d/dt (symm ∇u)
        #     = E : (symm ∇) u + η : d/dt (symm ∇) u
        #     = E : (symm ∇) u + η : (symm ∇) du/dt
        #     = E : (symm ∇) u + η : (symm ∇) V
        #     = E : (symm ∇) u + τ E : (symm ∇) V
        #     =: ℒ(u) + τ ℒ(V)   [constitutive law]
        #
        #   ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b   [Eulerian linear momentum balance]

        # θ time integration
        #
        # Values at the θ-point inside the timestep, with the unknown field, for linear terms.
        # The mechanical subproblem has no nonlinear terms.
        U = (1 - θ) * u_n + θ * u
        V = (1 - θ) * v_n + θ * v

        # Eulerian rates, θ approximation
        dudt = (u - u_n) / dt  # ∂u/∂t
        dvdt = (v - v_n) / dt  # ∂v/∂t

        # External fields, at the θ-point inside the timestep.
        #
        # Note the temperature and its *material* derivative are treated as known, external field inputs.
        # To compute them, couple this solver weakly to `InternalEnergyBalance`; these are designed
        # to work together to solve a thermomechanical problem.
        T = (1 - θ) * T_n + θ * T_
        dTdt = (1 - θ) * dTdt_n + θ * dTdt_
        b = (1 - θ) * b_n + θ * b_

        # Cauchy stress `σ`, using unknown `u` and `v`
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `extrafeathers.pdes.eulerian_solid.SteadyStateEulerianSolidPrimal`.
        #
        # TODO: Add Neumann BC for `(n·∇)u`; we need a zero normal gradient
        # TODO: BC in the analysis of 3D printing. For details,
        # TODO: see comments in `extrafeathers.pdes.eulerian_solid.SteadyStateEulerianSolidPrimal`.
        #
        def cauchy_stress(u, v, T, dTdt):
            """Form an expression representing the Cauchy stress."""
            # Note this is a closure that depends on the material parameters.
            λ = self.λ(T)
            μ = self.μ(T)
            α = self.α(T)
            dαdT = self.dαdT(T)

            Id = Identity(ε(u).geometric_dimension())
            K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

            Σ = K_inner(ε(u)) - K_inner(α) * (T - T0)  # elastic and elastothermal parts
            if self.τ > 0.0:  # viscous and viscothermal parts
                # Note this is nonlinear because of the term with `T * dTdt`;
                # fortunately, in the mechanical subproblem, `T` and `dTdt`
                # are known fields.
                Σ += τ * (K_inner(ε(v)) - (K_inner(α) +
                                           K_inner(dαdT) * (T - T0)) * dTdt)
            return Σ

        # For equation of `u`, θ-point value of σ, using the unknown fields.
        Σ = cauchy_stress(U, V, T, dTdt)

        # For visualization/export only, end-of-timestep value of σ, using end-of-timestep known values
        # (usable after solving the new `u` and `v` first).
        Σ_ = cauchy_stress(u_, v_, T_, dTdt_)

        # Set up the equations.
        #
        # Note that in the linear momentum balance law, `a` is the axial drive velocity field,
        # which is always divergence-free, so we may use the default mode (divergence-free) of
        # the `advw` and `advs` skew-symmetric advection operators.
        #
        # Because the Cauchy stress tensor is symmetric, we can symmetrize the test part of
        # the stress term:  ∇w → symm(∇w).
        F_u = (ρ * (dot(dvdt, w) * dx + advw(a, V, w, n)) +         # rate terms (axially comoving rate of `v`)
               inner(Σ.T, ε(w)) * dx -                              # stress
               ρ * dot(b, w) * dx)                                  # body force
        # Neumann BC for stress [Pa]. It is specified as the value of the Cauchy stress tensor,
        # which is then automatically projected into the direction of the outer unit normal.
        for Σ0, tag in self.bcσ:
            if tag is None:  # not specified -> whole Neumann boundary (i.e. everywhere that has no Dirichlet BC)
                F_u -= dot(dot(n, Σ0.T), w) * ds
            else:  # a specific part of the Neumann boundary
                if self.ds is None:
                    raise ValueError("`boundary_parts` must be supplied to build `ds` when Neumann BCs are applied on individual boundary parts.")
                F_u -= dot(dot(n, Σ0.T), w) * self.ds(tag)

        if self.V0 != 0.0:  # axial motion enabled?
            # SUPG stabilization for `u`. τ_SUPG as in an advection-diffusion problem (or as in Navier-Stokes).
            # Residual evaluated using values at the end of the timestep.
            # In practice, SUPG-stabilizing this equation doesn't seem to actually do much.
            deg = Constant(self.V.ufl_element().degree())
            λ = self.λ(T)
            μ = self.μ(T)
            moo = Maxx(λ, 2 * μ, τ * λ, τ * 2 * μ)  # representative diffusivity (note this is now a FEM field)
            τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (moo / ρ) / he**2)**-1  # [τ] = s
            R = (ρ * (dvdt + advs(a, v)) - div(Σ_) - ρ * b)
            F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, w), R) * dx
            F_u += F_SUPG

            # Velocity variable.
            #
            # There are a few important technical points here:
            #   - In terms of `u`, the linear momentum balance is second order in time. To use standard time integrators,
            #     we need to define some auxiliary velocity variable `v`, to transform the equation into a first-order system.
            #   - It is useful to choose `v` not as the local Eulerian rate of displacement in the laboratory, `∂u/∂t`, but instead
            #     as the co-moving derivative `du/dt`. Unlike the local Eulerian rate, this is a physically meaningful quantity:
            #     up to first order in the small quantities, it is the actual physical velocity of the material parcels,
            #     as measured against the co-moving frame.
            #   - Inserting this `v` into the Kelvin-Voigt constitutive law, we avoid the need for the extra space derivative
            #     that otherwise arises in the viscous terms when the material is undergoing axial motion. This allows us to
            #     apply standard C0-continuous FEM to an axially moving Kelvin-Voigt problem.
            #
            # (A programmer would say that the appearance of the extra space derivative, in the classical approach, hints of an
            #  /impedance mismatch/ (in the software engineering sense) between the classical auxiliary variable ∂u/∂t and the
            #  constitutive law being treated. In hindsight, it is obvious that the co-moving rate is a better choice, because
            #  the Kelvin-Voigt law explicitly talks about the co-moving rate of `u`. By representing that co-moving rate
            #  explicitly, the impedance mismatch goes away.)
            #
            # This equation essentially just defines `v` as the axially comoving rate of `u`. Keep in mind the MLE representation;
            # `u` is parameterized by the laboratory coordinate `x`, but measures displacement from a state with constant-velocity
            # axial motion at velocity `a`.
            #
            # We use skew-symmetric advection for numerical stabilization. Since `a` is the (divergence-free) axial drive
            # velocity field, we may use the default mode of `advw` and `advs`.
            F_v = (dot(V, ψ) * dx -
                   (dot(dudt, ψ) * dx + advw(a, U, ψ, n)))

            # SUPG stabilization for `v`. τ_SUPG as in a pure advection problem.
            # Residual evaluated using values at the end of the timestep.
            # Here the stabilization helps.
            deg = Constant(self.V.ufl_element().degree())
            τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he)**-1  # [τ] = s
            R = (v - (dudt + advs(a, u)))
            F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
            F_v += F_SUPG
        else:  # V0 == 0.0:
            # No axial motion, so the comoving rate reduces to the Eulerian rate.
            # We can drop the advection operator, and skip the SUPG stabilization, since it's not needed.
            F_v = (dot(V, ψ) * dx -
                   dot(dudt, ψ) * dx)

        # Assemble the equations
        F = F_u + F_v
        self.aform = lhs(F)
        self.Lform = rhs(F)

        # Visualization/export.
        # These become usable after solving the new `u_` and `v_` first.
        #
        # Cauchy stress.
        # This is an L2 projection of the primal representation, into tensor function space `Q`.
        F_σ = (inner(σ, φ) * dx -
               inner(Σ_, sym(φ)) * dx)
        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)

        # Strain.
        # This is an L2 projection of the primal representation, into tensor function space `P`.
        F_εu = inner(εu, q) * dx - inner(ε(u_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)

        # Strain rate (at material parcel).
        # Similarly.
        F_εv = inner(εv, q) * dx - inner(ε(v_), sym(q)) * dx
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

    def step(self) -> None:
        """Take a timestep of length `self.dt`.

        Stores the previous Picard iterate in `self.s_prev` (for convergence monitoring by user),
        and updates the Picard iterate `self.s_`. The algorithm used for solving the linear equation
        system is MUMPS (MUltifrontal Massively Parallel Sparse direct Solver).

        Can be called several times to iteratively refine the nonlinear solution.

        No return value.
        """
        begin("Linear momentum balance")
        self.s_prev.assign(self.s_)  # store previous Picard iterate
        A = assemble(self.aform)
        b = assemble(self.Lform)
        [bc.apply(A) for bc in self.bcu]
        [bc.apply(b) for bc in self.bcu]
        solve(A, self.s_.vector(), b, 'mumps')
        end()

    def export_strain(self) -> None:
        """Compute small-strain and strain rate tensors for visualization and/or export to other models.

        No return value; ε is written to `self.εu_`, and dε/dt is written to `self.εv_`.

        Note the *material* derivative; dε/dt is the strain rate experienced by the material parcels
        (not just an Eulerian rate).

        The fields are represented as FEM functions in the space `self.P` (see `__init__`).
        """
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')

        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

    def export_stress(self) -> None:
        """Compute Cauchy stress tensor for visualization and/or export to other models.

        No return value; σ is written to `self.σ_`.

        The field is represented as a FEM function in the space `self.Q` (see `__init__`).
        """
        A2 = assemble(self.a_σ)
        b2 = assemble(self.L_σ)
        solve(A2, self.σ_.vector(), b2, 'bicgstab', 'sor')

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        Copies `self.s_` to `self.s_n`, making the latest computed solution
        the "old" solution for the next timestep. The old "old" solution is discarded.

        Copies the latest thermal fields `self.T_` and `self.dTdt_` to
        `self.T_n` and `self.dTdt_n`, making them the "old" thermal fields
        for the next timestep. The old "old" thermal fields are discarded.

        Resets the old Picard iterate `self.s_prev`.

        Computes the stress and strain fields, making them available for
        visualization purposes. See `.export_stress()` and `.export_strain()`.

        No return value.
        """
        self.export_strain()
        self.export_stress()

        self.s_n.assign(self.s_)
        # Initial Picard iterate `s_` for new timestep is the final old value;
        # for that, we don't need to do anything.
        # The final old value is also the initial old Picard iterate `s_prev`.
        self.s_prev.assign(self.s_)

        # Initialize next old external fields.
        self.T_n.assign(self.T_)
        self.dTdt_n.assign(self.dTdt_)
        self.b_n.assign(self.b_)

# --------------------------------------------------------------------------------

class InternalEnergyBalance:
    """Eulerian internal energy balance law for a general anisotropic moving continuum.

    (A.k.a. the first law of thermodynamics for a continuum.)

    Keep in mind the heat equation is a parabolic problem. Thus, roughly speaking, to make
    the implicit midpoint rule (IMR) von Neumann stable, this solver needs  Δt ∝ he².
    This is unlike the linear momentum solver; linear momentum is a hyperbolic problem,
    which requires only  Δt ∝ he  for von Neumann stability. One possible solution is to
    use backward Euler (by setting θ = 1), which is A-stable, but has only O(Δt) accuracy.

    If `∂c/∂T ≠ 0`, and/or if `c` or `k` depend on temperature, then the PDE is nonlinear.

    This solver supports an iterative Picard linearization scheme. Calling `.step()` computes
    the next Picard iterate. It can be called several times to iteratively refine the solution.
    Call `.commit()` only after you are satisfied with the result for the current timestep.
    The previous Picard iterate is available in `self.s_prev`, for convergence monitoring by user.

    Also note that when the stress field is enabled (for the mechanical work contribution),
    the stress is typically a function of the temperature for thermomechanical reasons.
    So you may need to compute and send in a new `σ` (using the latest temperature field)
    after each Picard iteration. You can use the `LinearMomentumBalance` solver to do that.

    This equation is written purely in the *laboratory* frame; the mixed-Lagrangean-Eulerian
    (MLE) representation only applies to the linear momentum balance.

    Parameters:
        `V`: scalar function space. One copy is used for `u` and one for `v`.
        `ρ`: density [kg/m³]
        `c`: specific heat capacity [J / (kg K)]; temperature-dependent, callable: T -> Expression
             Scalar.
        `dcdT`: ∂c/∂T [J / (kg K²)]; temperature-dependent, callable: T -> Expression
                Scalar.
        `k`: heat conductivity [W / (m K)]; temperature-dependent, callable: T -> Expression
             Rank-2 tensor. For isotropic material, use  k' * Identity(2),  where k' is the
             scalar heat conductivity.
        `T0`: reference temperature for thermal expansion, at which
              the thermal expansion effect is considered zero [K]

              The value is stored as a UFL `Constant` in `self._T0`; if needed, this can be
              used in the expressions for the temperature-dependent parameters.

              To make it harder to create trivial bugs that may be difficult to track down,
              this parameter is mandatory. It is used for initializing the temperature field
              to a physically sensible value, so that this subproblem won't accidentally start
              at zero Kelvin.

              (The `LinearMomentumBalance` solver requires `T0` in any case, so forgetting to
               initialize the temperature field for that subproblem is much less likely; and
               these two solvers are meant to work together to solve the same thermomechanical
               problem.)

              The automatic initialization to `T0` is done for both the old value (initial condition),
              as well as the latest Picard iterate (initial guess for the first new value).
              The corresponding material derivative fields `dT/dt` are initialized to zero.

              If you want to use some other initial condition, do this after instantiating
              the solver::

                  my_initial_T = project(..., V)     # <-- your scalar field here
                  my_initial_dTdt = project(..., V)  # <-- your scalar field here
                  assigner = FunctionAssigner(S, [V, V])  # FunctionAssigner(receiving_space, assigning_space)
                  assigner.assign(solver.s_n, [my_initial_T, my_initial_dTdt])  # old value: the actual initial condition
                  solver.s_.assign(solver.s_n)  # latest Picard iterate: initial guess for new value

        `bcT`: Dirichlet boundary conditions for `u` [K]
        `bcq`: Neumann boundary conditions for heat flux `n·(k·∇u)` [W/m²]

               The Neumann boundary is defined as that part of the domain boundary
               for which no Dirichlet BCs for `u` have been given.

               Any part of the Neumann boundary for which a Neumann BC expression
               is not given, defaults to zero Neumann (i.e. perfect thermal insulator).

               The format is `[(fenics_expression, boundary_tag or None), ...]` where
               `None` means to use the expression on the whole Neumann boundary.

               A `boundary_tag` is an `int` that matches a boundary number in
               `boundary_parts`, which see. This means to use the expression
               on the specified part of the Neumann boundary only.

               IMPORTANT: the Neumann BC expressions are compiled just once, when the solver
               is instantiated. If you need to update a coefficient inside a Neumann BC
               during a simulation, give your `Expression` some updatable parameters::

                   heat_flux_strength = lambda t: 1e3 * min(t, 1.0)  # for example
                   heat_flux_neumann_bc = Expression("q0",
                                                     degree=1,
                                                     q0=heat_flux_strength(0.0))

                   # ...much later, during timestep loop...

                   heat_flux_neumann_bc.q0 = heat_flux_strength(t)

               And if you prefer to populate the Neumann BC list after instantiation
               (passing in an empty list initially, then adding BCs to it later),
               call `.compile_forms()` to refresh the equations to use the latest
               definitions.

        `dt`: timestep size [s]
        `θ`: parameter of the theta time integrator.

        `boundary_parts`: A facet `MeshFunction` that numbers the physical boundaries in the problem
                          (such as left edge, inlet, etc.).

                          This is the same `boundary_parts` that is used (in the main program of
                          a solver) to specify which boundaries Dirichlet BCs apply to; see the
                          example solvers. Produced by mesh generation with a suitable setup.

        `advection`, `velocity_degree`, `use_stress`, `stress_degree`:
            Same as in `extrafeathers.pdes.advection_diffusion.AdvectionDiffusion`.

    External field inputs:
        All these fields live on a vector/tensor element (as appropriate) of the same kind as `V`.
        By default (if `velocity_degree`, `stress_degree` are not given), the element uses the
        same degree as `V`.

        The degree must match the data you are sending in!

        Use `solver.xxx.assign(...)` to send in field values, where `xxx` is `a_` (new velocity),
        `a_n` (old velocity), `σ_`, or `σ_n`, as appropriate.

        `a`: Advection velocity field [m/s]. **Full velocity in the laboratory frame**,
             not just the velocity of material parcels in the co-moving frame.

             Used only if `advection != "off"`.

        `σ`: Cauchy stress tensor field [Pa], for mechanical work contribution.
             The term is  -σ : ∇a,  so it appears only when 1) advection is present,
             and 2) the velocity field `a` is non-uniform.

             Because `σ` is symmetric, we have  -σ : ∇a = -symm(σ) : ∇a = -σ : symm(∇a),
             so only the symmetric part of  ∇a  actually contributes to the mechanical work.
             This solver actually performs this symmetrization internally, to improve the
             numerical representation of the mechanical work term.

             Used only if `use_stress and advection != "off"`.
    """
    def __init__(self, V: FunctionSpace,
                 ρ: float,
                 c: typing.Callable, dcdT: typing.Callable,
                 k: typing.Callable, T0: float,
                 bcT: typing.List[DirichletBC],
                 bcq: typing.List[typing.Tuple[Expression, typing.Optional[int]]],
                 dt: float, θ: float = 0.5, *,
                 boundary_parts: typing.Optional[MeshFunction] = None,
                 advection: str = "general",
                 velocity_degree: int = None,
                 use_stress: bool = False,
                 stress_degree: int = None):
        if advection not in ("off", "divergence-free", "general"):
            raise ValueError(f"`advection` must be one of 'off', 'divergence-free', 'general'; got {type(advection)} with value {advection}")

        self.mesh = V.mesh()

        # Set up subdomains for the boundary parts.
        self.boundary_parts = boundary_parts
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_parts) if boundary_parts is not None else None

        # The stress term of the mechanical subproblem needs the material
        # derivative of the temperature field, so we introduce that as an
        # auxiliary variable, to be able to easily compute it in a numerically
        # stable way.
        #
        # This is similar to how we introduced the material parcel velocity
        # (as measured against the co-moving frame) as an auxiliary variable
        # for the linear momentum balance, but now we have a *material* derivative,
        # where the relevant velocity field is the velocity as measured against
        # the *laboratory* frame.
        #
        # Notation:
        #  u := temperature [K]
        #  v := material derivative of temperature [K/s]
        e = MixedElement(V.ufl_element(), V.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        w, ψ = TestFunctions(S)
        s_ = Function(S)  # suffix _: latest computed approximation
        u_, v_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, v_ = s_.sub(0), s_.sub(1)  # if you want the `Function` (for plotting etc.)
        s_n = Function(S)  # suffix _n: old value (end of previous timestep)
        u_n, v_n = split(s_n)

        # Previous Picard iterate, for convergence monitoring by user
        s_prev = Function(S)

        # Initialize T and dT/dt
        #
        # Each call to `.sub(j)` of a `Function` on a `MixedElement` seems to create a new copy.
        # We need `FunctionAssigner` to set values on the original `Function`, so that the field
        # does not vanish into a copy that is not used by the solver.
        assigner = FunctionAssigner(S, [V, V])  # FunctionAssigner(receiving_space, assigning_space)
        T0V = project(Constant(T0), V)
        zeroV = Function(V)  # filled by zeros by default
        assigner.assign(s_n, [T0V, zeroV])  # Old value (end of previous timestep; initial condition)
        assigner.assign(s_, [T0V, zeroV])  # New value for iterative solution
        assigner.assign(s_prev, [T0V, zeroV])  # Previous Picard iterate, for convergence monitoring by user

        self.u, self.v = u, v  # trials
        self.w, self.ψ = w, ψ  # tests
        self.u_, self.v_ = u_, v_  # latest computed approximation (UFL terms)
        self.u_n, self.v_n = u_n, v_n  # old value (end of previous timestep) (UFL terms)

        # Whole mixed scalar function space
        self.S = S
        self.s_ = s_
        self.s_n = s_n
        self.s_prev = s_prev

        # Original scalar function space, used for building the mixed scalar function space
        self.V = V

        # Vector and tensor variants of the original function space
        a_degree = velocity_degree if velocity_degree is not None else V.ufl_element().degree()
        σ_degree = stress_degree if stress_degree is not None else V.ufl_element().degree()
        V_rank1 = VectorFunctionSpace(self.mesh, V.ufl_element().family(), a_degree)
        V_rank2 = TensorFunctionSpace(self.mesh, V.ufl_element().family(), σ_degree)

        # Dirichlet boundary conditions
        self.bcT = bcT

        # Neumann BC for heat flux
        self.bcq = bcq

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Material parameters.
        # Single-argument functions: T -> Expression.
        self.k = k  # thermal conductivity [W / (m K)]
        self.c = c  # specific heat capacity [J / (kg K)]
        self.dcdT = dcdT  # ∂c/∂T

        # Advection velocity [m/s]. FEM function for maximum generality.
        # θ integration, so we need both old and new values for this too.
        self.advection = advection
        self.a_ = Function(V_rank1)
        self.a_.vector()[:] = 0.0
        self.a_n = Function(V_rank1)
        self.a_n.vector()[:] = 0.0

        # Stress [Pa]. FEM function for maximum generality.
        # θ integration, so we need both old and new values for this too.
        # Note the term is  -σ : ∇a,  so it only appears when advection is enabled.
        self.use_stress = use_stress and self.advection != "off"
        self.σ_ = Function(V_rank2)
        self.σ_.vector()[:] = 0.0
        self.σ_n = Function(V_rank2)
        self.σ_n.vector()[:] = 0.0

        # External specific heat source [W/kg]. FEM function for maximum generality.
        # Note the actual source term is  ρ h  [W/m³]; the solver automatically multiplies by ρ.
        self.h_ = Function(V)
        self.h_.vector()[:] = 0.0
        self.h_n = Function(V)
        self.h_n.vector()[:] = 0.0

        # Constant parameters.
        self._ρ = Constant(ρ)
        self._T0 = Constant(T0)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = StabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1 / self.V.ufl_element().degree())

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    T0 = ufl_constant_property("T0", doc="Reference temperature for thermal expansion [K]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Temperature
        u = self.u  # new (unknown)
        u_ = self.u_  # latest available (TODO: should initialize this to `u_n` before first timestep)
        w = self.w  # test
        u_n = self.u_n  # old (end of previous timestep)

        # Material derivative of temperature
        v = self.v
        v_ = self.v_
        ψ = self.ψ
        v_n = self.v_n

        # Convection velocity
        a_ = self.a_
        a_n = self.a_n

        # Stress
        σ_ = self.σ_
        σ_n = self.σ_n

        # Specific heat source
        h_ = self.h_
        h_n = self.h_n

        # Local mesh size (for stabilization terms)
        he = self.he

        # Constant parameters
        ρ = self._ρ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # θ time integration
        #
        # Values at the θ-point inside the timestep, with the unknown field, for linear terms.
        U = (1 - θ) * u_n + θ * u
        V = (1 - θ) * v_n + θ * v

        # Values at the θ-point inside the timestep, with the latest iterate, for nonlinear terms.
        U_ = (1 - θ) * u_n + θ * u_
        V_ = (1 - θ) * v_n + θ * v_

        # Values at the θ-point inside the timestep, for external field inputs.
        a = (1 - θ) * a_n + θ * a_
        σ = (1 - θ) * σ_n + θ * σ_
        h = (1 - θ) * h_n + θ * h_

        # Eulerian rates, θ approximation
        dudt = (u - u_n) / dt  # ∂u/∂t
        # Unlike in the linear momentum balance, here ∂v/∂t is not needed,
        # because the internal energy balance is first order in time.

        # Evaluate the parameters at the temperature at the θ-point inside the timestep.
        # We need explicit values, and any dependence on temperature in these parameters
        # makes the equation nonlinear, so we use the latest-iterate version `U_`.
        c = self.c(U_)
        dcdT = self.dcdT(U_)
        k = self.k(U_)

        # Equation of temperature, `u` [K]
        #
        # We only linearize; it is the caller's responsibility to iterate.
        #
        # Note that due to the splitting to `u` and `v`, the equation system is not symmetric.
        #  - It is a 2×2 block system.
        #  - The heat conduction term  ∫ [k·∇u]·∇w dx   lives in the upper left block (u, w).
        #    This term is symmetric, because `k` is a symmetric rank-2 tensor.
        #  - The rate term  ∫ ρ c v w dx  lives in the upper right block (v, w).
        #  - The rate term  ∫ ρ dc/dT v u w dx  distributes into both upper blocks,
        #    via the symmetric linearization of "v u".
        UV = 0.5 * (U * V_ + U_ * V)                             # symmetric linearization
        F_u = (ρ * (dcdT * UV + c * V) * w * dx +                # rate terms
               dot(dot(k, nabla_grad(U)), nabla_grad(w)) * dx -  # heat conduction
               ρ * h * w * dx)                                   # external heat source
        # Neumann BC for heat flux.
        #
        # The Neumann BC is given as the scalar heat flux [W/m²] in the direction of the outer unit normal:
        #   q0 = dot(n, dot(k, nabla_grad(U))) on the Neumann boundary
        # An alternative (that we don't use here) would be to specify the flux vector k·∇u.
        for q0, tag in self.bcq:
            if tag is None:  # not specified -> whole Neumann boundary (i.e. everywhere that has no Dirichlet BC)
                F_u -= q0 * w * ds
            else:  # a specific part of the Neumann boundary
                if self.ds is None:
                    raise ValueError("`boundary_parts` must be supplied to build `ds` when Neumann BCs are applied on individual boundary parts.")
                F_u -= q0 * w * self.ds(tag)
        if self.use_stress and self.advection != "off":
            # Because the Cauchy stress tensor is symmetric, we can symmetrize the test part of
            # the mechanical work term:  ∇a → symm(∇a).
            F_u -= inner(σ, ε(a)) * w * dx                       # mechanical work

        # Equation of material derivative of temperature, `v` [K/s]
        #   v := dudt = ∂u/∂t + (a·∇) u
        # solved essentially as an L2 projection of the right-hand side.
        if self.advection != "off":
            assert self.advection in ("divergence-free", "general")

            # The equation of `u` has no advection terms, so it needs no stabilization.

            # Material rate. This equation essentially just defines `v` as the material rate of `u`.
            #
            # Splitting this into a separate equation isolates the numerically tricky
            # advection effect, allowing us to easily apply stabilization techniques,
            # such as SUPG. It also shortens the equation of `u` quite a bit, and makes
            # its form much closer to the classical case with no axial motion.
            #
            # Note the `V` term lives in the lower-right block (v, ψ), which itself is symmetric,
            # but the coupling terms live in the lower-left block (u, ψ).
            #
            # We use skew-symmetric advection for numerical stabilization.
            F_v = (dot(V, ψ) * dx -
                   (dot(dudt, ψ) * dx + advw(a, U, ψ, n, mode=self.advection)))

            # SUPG stabilization for `v`. τ_SUPG as in a pure advection problem.
            # Residual evaluated using values at the end of the timestep.
            # Here the stabilization helps.
            deg = Constant(self.V.ufl_element().degree())
            τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a_) / he)**-1  # [τ] = s
            R = (v - (dudt + advs(a_, u, mode=self.advection)))
            F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a_, ψ, mode=self.advection), R) * dx
            F_v += F_SUPG
        else:
            # No advection, so the material rate reduces to the Eulerian rate.
            # In this case, the lower-left block (u, ψ) is itself symmetric,
            # as is the lower-right block (v, ψ).
            F_v = (dot(V, ψ) * dx -
                   dot(dudt, ψ) * dx)

        # Assemble the equations
        F = F_u + F_v
        self.aform = lhs(F)
        self.Lform = rhs(F)

    def step(self) -> int:
        """Take a timestep of length `self.dt`.

        Stores the previous Picard iterate in `self.s_prev` (for convergence monitoring by user),
        and updates the Picard iterate `self.s_`. The algorithm used for solving the linear equation
        system is BiCGstab, with the hypre_amg preconditioner.

        Can be called several times to iteratively refine the nonlinear solution.

        Returns the number of Krylov iterations taken.
        """
        begin("Internal energy balance")
        self.s_prev.assign(self.s_)  # store previous Picard iterate
        A = assemble(self.aform)
        b = assemble(self.Lform)
        [bc.apply(A) for bc in self.bcT]
        [bc.apply(b) for bc in self.bcT]
        solve(A, self.s_.vector(), b, 'mumps')
        it = 1
        # it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')
        end()
        return it

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        Call this after you are satisfied with the result of the current timestep.

        Copies `self.s_` to `self.s_n`, making the latest computed solution
        the "old" solution for the next timestep. The old "old" solution is discarded.

        Copies the latest external fields `self.σ_` and `self.a_` to
        `self.σ_n` and `self.a_n`, making them the "old" external fields
        for the next timestep. The old "old" external fields are discarded.

        Resets the old Picard iterate `self.s_prev`.

        No return value.
        """
        self.s_n.assign(self.s_)
        # Initial Picard iterate `s_` for new timestep is the final old value;
        # for that, we don't need to do anything.
        # The final old value is also the initial old Picard iterate `s_prev`.
        self.s_prev.assign(self.s_)

        # Initialize next old external fields.
        self.σ_n.assign(self.σ_)
        self.a_n.assign(self.a_)
        self.h_n.assign(self.h_)
