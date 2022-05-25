# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of uniform axial motion).

Three alternative formulations are provided, both in dynamic and in steady-state cases:

 - `EulerianSolid`, `SteadyStateEulerianSolid`:

   Straightforward Eulerian description. Variables are `u(x, t)`, `v(x, t) := ∂u/∂t`, and `σ(x, t)`.

   `v` is the Eulerian rate of `u`.

 - `EulerianSolidAlternative`, `SteadyStateEulerianSolidAlternative`:

   Eulerian description using material parcel velocity. Variables are `u(x, t)`, `v(x, t) := du/dt`,
   and `σ(x, t)`.

   `v` is the material derivative of `u`; it is the actual physical velocity of the material
   parcels with respect to the co-moving frame.

   Note `v` is still an Eulerian field; it is a spatial description of a material quantity!

 - `EulerianSolidPrimal`, `SteadyStateEulerianSolidPrimal`:

   Eulerian description using material parcel velocity and primal variables only. Variables are
   `u(x, t)`, and `v(x, t) := du/dt`.

   `v` is the material derivative of `u`; it is the actual physical velocity of the material
   parcels with respect to the co-moving frame.

   Note `v` is still an Eulerian field; it is a spatial description of a material quantity!

   This is the cleanest formulation, and the fastest solver.

**NOTE**:

Of the steady-state solvers, currently only `SteadyStateEulerianSolidPrimal` converges to the
correct solution; this is still something to be investigated later. For now, if you want the
steady state, just use `SteadyStateEulerianSolidPrimal`.

All three dynamic solvers work as expected.
"""

__all__ = ["EulerianSolid",
           "SteadyStateEulerianSolid",  # does not work yet
           "EulerianSolidAlternative",
           "SteadyStateEulerianSolidAlternative",  # does not work yet
           "EulerianSolidPrimal",
           "SteadyStateEulerianSolidPrimal",
           "step_adaptive"]

from contextlib import contextmanager
import typing

from fenics import (VectorFunctionSpace, TensorFunctionSpace,
                    MixedElement, FunctionSpace, TrialFunctions, TestFunctions, split, FunctionAssigner, project,
                    TrialFunction, TestFunction,
                    Constant, Expression, Function,
                    FacetNormal, DirichletBC,
                    dot, inner, outer, sym, tr,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve,
                    interpolate, VectorSpaceBasis, as_backend_type,
                    norm,
                    begin, end)

from ..meshfunction import meshsize, cell_mf_to_expression
from .numutil import ε, mag, advw, advs
from .util import ufl_constant_property, StabilizerFlags


def null_space_fields(dim):
    """Set up null space for removal in the Krylov solver.

    Return a `list` of rigid-body modes of geometric dimension `dim` as FEniCS
    expressions. These can then be projected into the correct finite element space.

    Null space of the linear momentum balance is {u: ε(u) = 0 and ∇·u = 0}
    This consists of rigid-body translations and infinitesimal rigid-body rotations.

    Strictly, this is the null space of linear elasticity, but the physics shouldn't
    be that much different for the other linear models.

    See:
        https://fenicsproject.discourse.group/t/rotation-in-null-space-for-elasticity/4083
         https://bitbucket.org/fenics-project/dolfin/src/946dbd3e268dc20c64778eb5b734941ca5c343e5/python/demo/undocumented/elasticity/demo_elasticity.py#lines-35:52
        https://bitbucket.org/fenics-project/dolfin/issues/587/functionassigner-does-not-always-call
    """
    if dim == 1:
        fus = [Constant(1)]
    elif dim == 2:
        fus = [Constant((1, 0)),
               Constant((0, 1)),
               Expression(("x[1]", "-x[0]"), degree=1)]  # around z axis (clockwise)
    elif dim == 3:
        fus = [Constant((1, 0, 0)),
               Constant((0, 1, 0)),
               Constant((0, 0, 1)),
               Expression(("0", "x[2]", "-x[1]"), degree=1),  # around x axis (clockwise)
               Expression(("-x[2]", "0", "x[0]"), degree=1),  # around y axis (clockwise)
               Expression(("x[1]", "-x[0]", "0"), degree=1)]  # around z axis (clockwise)
    else:
        raise NotImplementedError(f"dim = {dim}")
    return fus


class EulerianSolidStabilizerFlags(StabilizerFlags):
    """Interface for numerical stabilizer on/off flags.

    Collects them into one namespace; handles translation between
    `bool` values and the UFL expressions that are actually used
    in the equations.

    Usage::

        print(solver.stabilizers)  # status --> "<EulerianSolidStabilizerFlags: SUPG(True)>"
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


# TODO: use nondimensional form
class EulerianSolid:
    """Axially moving linear solid, small-displacement Eulerian formulation.

    For now, this solver provides the linear elastic and Kelvin-Voigt models.

    The spatial discretization is based on a mixed formulation. For linear
    viscoelastic models, the axial motion introduces a third derivative in the
    strong form of the primal formulation. Therefore, the primal formulation
    requires C1 elements (not available in FEniCS, for mathematical reasons
    outlined in Kirby & Mitchell, 2019; this was done manually for an axially
    moving sheet in Kurki et al., 2016).

    The alternative, chosen here, is a mixed formulation where both `u` and `σ`
    appear as unknowns. The additional derivative from the axial motion then
    appears as a spatial derivative of ε in the constitutive equation for σ.

    Time integration is performed using the θ method; Crank-Nicolson by default.

    `V`: vector function space for displacement
    `Q`: tensor function space for stress
    `P`: tensor function space for strain projection
         Strains are L2-projected into `P` before using them in the constitutive
         law. Improves stability in Kelvin-Voigt in the presence of axial motion.
    `ρ`: density [kg / m³]
    `λ`: Lamé's first parameter [Pa]
    `μ`: shear modulus [Pa]
    `τ`: Kelvin-Voigt retardation time [s].

         Defined as `τ := η / E`, where `E` is Young's modulus [Pa], and
         `η` is the viscous modulus [Pa s].

         **CAUTION**: The initial value of `τ` passed in to the constructor
         determines the material model.

         If you pass in `τ=...` with a nonzero value, the solver will
         set up the PDEs for the Kelvin-Voigt model.

         If you pass in `τ=0`, the solver will set up the PDEs for the
         linear elastic model.

         Setting the value of `τ` later (`solver.τ = ...`) does **not**
         affect which model is in use; so if needed, you can perform
         studies with Kelvin-Voigt where `τ` changes quasistatically.

         To force a model change later, set the new value of `τ` first,
         and then call the `compile_forms` method to refresh the PDEs.

    `V0`: velocity of co-moving frame in +x direction (constant) [m/s]
    `bcv`: Dirichlet boundary conditions for Eulerian displacement rate ∂u/∂t.
           The displacement `u` takes no BCs; use an initial condition instead.
    `bcσ`: Dirichlet boundary conditions for stress (NOTE: must set only n·σ).
           Alternative for setting `v`.
    `dt`: timestep [s]
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 is Crank-Nicolson; 0 is forward Euler, 1 is backward Euler.

         Note that for θ = 0, the SUPG stabilization parameter τ_SUPG → 0,
         so when using forward Euler, it does not make sense to enable the
         SUPG stabilizer.

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.

    For LBB-condition-related reasons, the space `Q` must be much larger than `V`; both
    {V=Q1, Q=Q2} and {V=Q1, Q=Q3} have been tested to work and to yield similar results.

    Near-term future plans (when I next have time to work on this project) include
    extending this to support SLS (the standard linear solid); this should be a fairly
    minor modification, just replacing the equation for `σ` by a PDE; no infra changes.
    Far-future plans include a viscoplastic model of the Chaboche family (of which a
    Lagrangean formulation is available in the Julia package `Materials.jl`).


    **Equations**:

    The formulation used by `EulerianSolid` is perhaps the most straightforward
    Eulerian formulation for an axially moving continuum. The momentum balance
    for a continuum is

        ρ dV/dt - ∇·σ = ρ b

    where `V` is the material parcel velocity in an inertial frame of our
    choice (the law is postulated to be Galilean invariant).

    Let us choose the laboratory frame. We have

        dV/dt ≈ [∂²u/∂t² + 2 (a·∇) ∂u/∂t + (a·∇)(a·∇) u]

    Thus the Eulerian description of the momentum balance becomes

        ρ ∂²u/∂t² + 2 ρ (a·∇) ∂u/∂t + ρ (a·∇)(a·∇)u - ∇·σ = ρ b

    where, for Kelvin-Voigt (linear elastic as special case τ = 0):

        σ = E : ε + η : dε/dt
          = E : (symm ∇u) + η : d/dt (symm ∇u)
          = E : (symm ∇) u + η : d/dt (symm ∇) u
          = E : (symm ∇) u + η : (symm ∇) du/dt
          = E : (symm ∇) u + η : (symm ∇) (∂u/∂t + (a·∇)) u
          = E : (symm ∇) u + η : [(symm ∇) v + (a·∇) u]
          = E : (symm ∇) u + τ E : [(symm ∇) v + (symm ∇) (a·∇) u]
          = E : (symm ∇) u + τ E : [(symm ∇) v + (a·∇) (symm ∇) u]
          = E : ε + τ E : [(symm ∇) v + (a·∇) ε]

    and we have defined

        v := ∂u/∂t

    i.e. our velocity-like variable is the Eulerian rate of displacement.

    The displacement `u` is an Eulerian field, measured with respect to a
    reference state where the material travels at constant axial velocity.
    It is parameterized by the laboratory coordinate `x`.

    This formulation requires integration by parts on the right-hand side of the
    weak-form constitutive equation to get rid of the second derivative of `u`.

    Also, numerical stability is improved by a practical trick: we project
    `(symm ∇) u` and `(symm ∇) v` into a C0 continuous space before using
    the data in the constitutive equation. This allows us to actually integrate
    by parts in only "half" of the `(a·∇) ε` term, enabling the use of the
    numerically more stable skew-symmetric advection discretization.


    **References**:

        Robert C. Kirby and Lawrence Mitchell. 2019. Code generation for generally
        mapped finite elements. ACM Transactions on Mathematical Software 45(41):1-23.
        https://doi.org/10.1145/3361745
        https://arxiv.org/abs/1808.05513

        Matti Kurki, Juha Jeronen, Tytti Saksa, and Tero Tuovinen. 2016.
        The origin of in-plane stresses in axially moving orthotropic continua.
        International Journal of Solids and Structures 81, 43-62.
        https://doi.org/10.1016/j.ijsolstr.2015.10.027
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcv: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Trial and test functions
        #
        # `u`: displacement [new algorithm: algebraic update for `u`; no trial or test functions]
        # `v`: Eulerian time rate of displacement, i.e. Eulerian velocity in linear momentum balance
        # `σ`: stress
        #  - Mixed formulation; stress has its own equation to allow easily
        #    changing the constitutive model.
        #  - Also, need to treat it this way for an Eulerian description of
        #    viscoelastic models, because the material derivative introduces
        #    a term that is one order of ∇ higher. In the primal formulation,
        #    in the weak form, this requires taking second derivatives of `u`.

        # u = TrialFunction(V)  # no suffix: UFL symbol for unknown quantity
        # w = TestFunction(V)
        v = TrialFunction(V)  # no suffix: UFL symbol for unknown quantity
        ψ = TestFunction(V)
        σ = TrialFunction(Q)
        φ = TestFunction(Q)

        u_ = Function(V)  # suffix _: latest computed approximation
        u_n = Function(V)  # suffix _n: old value (end of previous timestep)
        v_ = Function(V)
        v_n = Function(V)
        σ_ = Function(Q)
        σ_n = Function(Q)

        self.V = V
        self.Q = Q

        # TODO: These were only used for manual patch-averaging; remove?
        self.VdG0 = VectorFunctionSpace(self.mesh, "DG", 0)  # "DG" is a handy alias for "DP or DQ dep. on mesh"
        self.QdG0 = TensorFunctionSpace(self.mesh, "DG", 0)

        self.P = P
        self.w = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.v, self.σ = v, σ  # trials
        self.ψ, self.φ = ψ, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # latest computed approximation
        self.u_n, self.v_n, self.σ_n = u_n, v_n, σ_n  # old value (end of previous timestep)

        # Set up the null space for removal in the Krylov solver.
        fus = null_space_fields(self.mesh.geometric_dimension())
        null_space_basis = [interpolate(fu, V).vector() for fu in fus]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcv = bcv
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        # PDE system iteration parameters.
        # User-configurable (`solver.maxit = ...`), but not a major advertised feature.
        self.maxit = 100  # maximum number of system iterations per timestep
        self.tol = 1e-8  # system iteration tolerance, ‖v - v_prev‖_H1 (over the whole domain)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u_ = self.u_    # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Eulerian time rate of displacement,  v = ∂u/∂t
        v = self.v      # new (unknown)
        ψ = self.ψ      # test
        v_ = self.v_    # latest available approximation
        v_n = self.v_n  # old (end of previous timestep)

        # Stress
        σ = self.σ
        φ = self.φ
        σ_ = self.σ_
        σ_n = self.σ_n

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem

        # # Step 1: ∂u/∂t = v -> obtain `u` (explicit in `v`)
        # #
        # # NOTE: We don't actually use this first equation, since we update `u` DOF-wise from `v`,
        # # essentially using the same equation, but with lumped mass. That yields a numerically
        # # more stable simulation. This old code is kept for documentation purposes only.
        # dudt = (u - u_n) / dt
        # V = (1 - θ) * v_n + θ * v_  # known; initially `v_ = v_n`, but this variant can be iterated.
        # # V = v_n  # forward Euler
        # # V = v_  # backward Euler (iterative)
        # F_u = dot(dudt - V, w) * dx

        # Step 2: solve `σ`, using `u` from step 1 (and if needed, the latest available `v`)
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.

        # θ integration: field values at the "θ-point" in time:
        U = (1 - θ) * u_n + θ * u_  # known
        V = (1 - θ) * v_n + θ * v_  # known
        Σ = (1 - θ) * σ_n + θ * σ   # unknown!
        # # Backward Euler integration:
        # U = u_
        # V = v_
        # Σ = σ

        # Step 1½: Project the strains into a C0 space - this makes them once differentiable.
        εu = self.εu  # unknown
        εv = self.εv
        w = self.w
        F_εu = inner(εu, w) * dx - inner(ε(U), sym(w)) * dx
        F_εv = inner(εv, w) * dx - inner(ε(V), sym(w)) * dx

        # After strain projection:
        εu_ = self.εu_  # known, at the "θ-point" in time
        εv_ = self.εv_
        Id = Identity(εu_.geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

        # # Original definitions for step 2, no projection
        # εu_ = ε(U)
        # εv_ = ε(V)
        # Id = Identity(εu_.geometric_dimension())
        # K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

        # Choose constitutive model
        if self.τ == 0.0:  # Linear elastic (LE)
            # Doesn't matter whether classical or axially moving,
            # since there are no time derivatives in the constitutive law.
            #
            # In general:
            #   σ = K : ε
            # With isotropic elastic symmetry, the stiffness tensor is:
            #   K = 2 μ ES + 3 λ EV
            # Because for any rank-2 tensor T, it holds that
            #   ES : T = symm(T)
            #   EV : T = vol(T) = (1/3) I tr(T)
            # we have
            #   σ = 2 μ symm(ε) + 3 λ vol(ε)
            #     = 2 μ ε + 3 λ vol(ε)
            #     = 2 μ ε + λ I tr(ε)
            # where on the second line we have used the symmetry of ε to drop the `symm(...)`.
            #
            # For LE, we obtain the stress by simply L2-projecting the above explicit expression
            # into the basis of `σ`.
            #
            # Note that the expression gives the stress at the "θ-point" in time; using our definition
            # of `Σ` above, FEniCS automatically sorts that out for us.
            F_σ = (inner(Σ, φ) * dx -
                   inner(K_inner(εu_), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            # σ = 2 [μ + μ_visc d/dt] ε + I tr([λ + λ_visc d/dt] ε)
            #   = 2 μ [1 + τ d/dt] ε + λ I tr([1 + τ d/dt] ε)
            #   = 2 μ [1 + τ (∂/∂t + a·∇)] ε + λ I tr([1 + τ (∂/∂t + a·∇)] ε)
            #   = 2 μ [1 + τ (∂/∂t + a·∇)] ε + λ I [tr(ε) + τ (tr(∂ε/∂t) + a·∇ tr(ε))]
            # where we have expressed the material derivative in its Eulerian representation,
            # `d/dt = ∂/∂t + a·∇`.
            #
            # In FEniCS, this would read as:
            #   εv = ε(V)
            #   stress_expr = (2 * μ * (εu + τ * (εv + advs(a, εu))) +
            #                  λ * I * (tr(εu) + τ * (tr(εv) + advs(a, tr(εu)))))
            #   F_σ = inner(Σ - stress_expr, φ) * dx
            #
            # But since we have C0 elements, the spatial derivatives of ε = symm(∇u) are not defined
            # across element boundaries (in the original algorithm, without strain projection into C0 space).
            # Thus we must move the advection operator to the test function. Equation (774) in the report
            # (section "Weak form of the constitutive law") gives the appropriate integration-by-parts formula:
            #
            #   ∫ (φ:Kη):(a·∇)ε dx = ∫ (a·n) (φ:Kη:ε) dΓ - ∫ [(a·∇)φ]:Kη:ε dx
            #
            # **CAUTION**: This is for a constant viscous stiffness tensor Kε; temperature dependence
            # of Kε will add more terms here.
            #
            # For Kelvin-Voigt, Kε = τ K, where K is the elastic stiffness tensor.
            #
            # We can symmetrize the test function on the RHS (like we do σ:∇ψ -> σ:(symm ∇ψ)
            # in the momentum equation), because the other operand of the double-dot product
            # (K:ε(u) or K:ε(v)) is symmetric in all terms here.
            #
            # # let's try streamline upwinding?
            # φup = φ + he / mag(a) * dot(a, nabla_grad(φ))
            #
            # TODO: Compute as projection into lower-degree space?  PΣ = ...
            # TODO: Doesn't seem to be supported by FEniCS. Let's just Σ = ...
            #
            # TODO: Find how to set BCs in domain interior (to fix displacement)
            # TODO: Debug the rigid-body mode eliminator
            # TODO: Debug the steady-state solver

            # # Original, integrated by parts.
            # F_σ = (inner(Σ, φ) * dx -
            #        (inner(K_inner(εu_) + τ * K_inner(εv_), sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
            #         τ * dot(a, n) * inner(K_inner(εu_), sym(φ)) * ds -  # generated by ∫ (φ:Kη):(a·∇)ε dx
            #         τ * inner(K_inner(εu_), advs(a, sym(φ))) * dx))  # generated by ∫ (φ:Kη):(a·∇)ε dx

            # # Look, ma, no integration by parts.
            # F_σ = (inner(Σ, φ) * dx -
            #        (inner(K_inner(εu_) + τ * K_inner(εv_), sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
            #         τ * inner(K_inner(advs(a, εu_)), sym(φ)) * dx))  # ∫ (φ:Kη):(a·∇)ε dx

            # "Symmetrized": half integrated by parts, half not.
            # The `a·∇` operates on both the quantity and test parts. This is most similar to the
            # skew-symmetric advection operator used in modern FEM discretizations of Navier-Stokes.
            F_σ = (inner(Σ, φ) * dx -
                   (inner(K_inner(εu_) + τ * K_inner(εv_), sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
                    0.5 * τ * inner(K_inner(advs(a, εu_)), sym(φ)) * dx -  # 1/2 ∫ (φ:Kη):(a·∇)ε dx
                    0.5 * τ * inner(K_inner(εu_), advs(a, sym(φ))) * dx +  # 1/2 ∫ (φ:Kη):(a·∇)ε dx
                    0.5 * τ * dot(a, n) * inner(K_inner(εu_), sym(φ)) * ds))  # generated, 1/2 ∫ (φ:Kη):(a·∇)ε dx

            # # TODO: Fix the stress stabilization. Commented out, because doesn't seem to work correctly.
            # #
            # # SUPG stabilization. Note that the SUPG terms are added only in the element interiors.
            # τ_SUPG = (α0 / self.Q.ufl_element().degree()) * (1 / (θ * dt) + 2 * mag(a) / he)**-1  # [τ] = s  # TODO: tune value
            # # τ_SUPG = (α0 / self.Q.ufl_element().degree()) * (2 * mag(a) / he)**-1  # [τ] = s  # TODO: tune value
            # # τ_SUPG = Constant(0.004)  # TODO: tune value
            # # The residual is evaluated elementwise in strong form, at the end of the timestep.
            # εu_ = ε(u_)  # at end of timestep
            # εv_ = ε(v_)
            # # # We need advs(a, εu_), but if `u` uses a degree-1 basis, ∇εu_ = 0. Thus, we need to
            # # # project εu_ into Q before differentiating to get a useful derivative in the element interiors.
            # # # Use a temporary storage (εtmp) for that; the iteration loop fills in the actual data.
            # # R = σ - (K_inner(εu_) + τ * (K_inner(εv_) + K_inner(advs(a, self.εtmp))))
            # # OTOH, in Navier-Stokes, some of the terms may be zero and SUPG works just fine.
            # R = σ - (K_inner(εu_) + τ * (K_inner(εv_) + K_inner(advs(a, εu_))))
            # # Same here; the RHS is symmetric, so symmetrize the test function.
            # F_SUPG = enable_SUPG_flag * τ_SUPG * inner(advs(a, sym(φ)), R) * dx
            # F_σ += F_SUPG

            # # Let's try classic artificial diffusion along streamlines?
            # #
            # # No idea about tuning factor; he²/mag(a)² has the right units (s², to cancel the units of (a·∇)²,
            # # so that the units of the form match the other terms of the equation), but its value seems too
            # # large, leading to excessive smoothing. Multiplying it by a further 0.1 seems to work, but
            # # no idea why.
            # #
            # # # DEBUG - for 16×16 quads, `he` seems to be ~0.0884
            # # # TODO: debug missing C++ `eval` method of our `cell_mf_to_expression`
            # # # TODO: (it does have one, but it needs a cell number; maybe need a variant without the cell)
            # # import matplotlib.pyplot as plt
            # # import dolfin
            # # from .. import plotmagic
            # # theplot = plotmagic.mpiplot(dolfin.project(he, self.V.sub(0).collapse()))
            # # if dolfin.MPI.comm_world.rank == 0:
            # #     plt.colorbar(theplot)
            # #     plt.show()
            # # from sys import exit
            # # exit(0)
            # #
            # # Qdeg = Constant(self.Q.ufl_element().degree())
            # tune = Constant(0.1)
            # F_σ += tune * (he**2 / mag(a)**2) * inner(dot(a, nabla_grad(σ)), dot(a, nabla_grad(sym(φ)))) * dx
            # # F_σ += θ * dt * (he / mag(a)) * inner(dot(a, nabla_grad(σ)), dot(a, nabla_grad(sym(φ)))) * dx

        # Step 3: solve `v` from momentum equation
        #
        # ------------------------------------------------------------
        # IMPORTANT NOTE on the formulation.
        #
        # Temporarily, for the sake of explanation, let us denote the material
        # parcel velocity in the co-moving frame (i.e. the true physical velocity
        # of the material, minus the uniform axial motion) by `V`:
        #
        #   V := du/dt
        #
        # Its Eulerian representation in the laboratory frame is
        #
        #   V = ∂u/∂t + (a·∇) u    (*)
        #
        # where `a` is the velocity of the co-moving frame, measured against the
        # laboratory frame, and `u` is the Eulerian displacement as measured in
        # the co-moving frame (i.e., displacement field parameterized by the
        # spatial coordinate, with its value measured with respect to the reference
        # state where the material moves uniformly at the axial velocity `a`).
        #
        # The material derivative (Lagrangean rate) operator `d/dt` itself has
        # the Eulerian representation
        #
        #   d/dt = ∂/∂t + (V·∇)
        #        ≈ ∂/∂t + (a·∇)    (**)
        #
        # where, on the second line, we have ignored the small motion of the
        # material with regard to the co-moving frame, `V ≈ a`, used only for
        # the purposes of approximating this operator in an easily evaluated
        # form that is first-order accurate.
        #
        # Observe that (*) is exact, but (**) is an approximation.
        #
        # (Keep in mind this is a model for the small-displacement regime, so when
        #  we apply this operator to a small quantity, the effects from that small
        #  motion will be second-order small; thus we may ignore them.)
        #
        # Now, recall the linear momentum balance for a general continuum:
        #
        #   ρ dV/dt - ∇·σ = ρ b
        #
        #
        # In the formulation used here (class `EulerianSolid`), we have defined
        # our velocity variable `v` as the Eulerian rate of `u`:
        #
        #   v := ∂u/∂t    (***)
        #
        # Thus we find the Eulerian representation of `dV/dt`, in the laboratory
        # frame, as
        #
        #   dV/dt = d/dt [(∂/∂t + (a·∇)) u]               [by (*)]
        #         ≈ (∂/∂t + (a·∇))² u                     [by (**)]
        #         = ∂²u/∂t² + 2 (a·∇) ∂u/∂t + (a·∇)² u   [expand]
        #         ≡ ∂v/∂t + 2 (a·∇) v + (a·∇)² u         [by (***)]
        #
        # which is what we have (converted into weak form) below.
        #
        # In this formulation, the linear momentum balance and the constitutive
        # law take nonstandard Eulerian forms that account for the uniform
        # axial motion, but `u` is simply the time integral of `v`.
        #
        #
        # An alternative formulation is implemented in class `EulerianSolidAlternative`.
        #
        # If we use `V` as our velocity variable instead of `v`, then we have,
        # in the co-moving frame (exactly, no approximation)
        #
        #   dV/dt = ∂V/∂t + (V·∇) V
        #
        # In the laboratory frame, the material parcel velocity is `a + V`:
        #
        #   d(a + V)/dt = [∂/∂t + ((a + V)·∇)] (a + V)
        #               = [∂/∂t + ((a + V)·∇)] V          [`a` is constant]
        #               ≈ [∂/∂t + (a·∇)] V                [drop 2nd-order small term]
        #
        # The linear momentum balance becomes very similar to that of Navier-Stokes,
        # albeit the constitutive equation is different (because this is a solid),
        # and to be able to evaluate the stress, we need to track the displacement `u`.
        #
        # The linear momentum balance and the constitutive equation take their
        # standard (not axially moving) forms, which are easier to handle
        # numerically, but then for updating `u`, we need to solve the linear
        # first-order transport PDE (*).
        #
        # For the axially moving Kelvin-Voigt material, the full set of equations
        # in the alternative formulation is:
        #
        #   ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b   [linear momentum balance]
        #
        #   σ = E : ε + η : dε/dt
        #     = E : (symm ∇u) + η : d/dt (symm ∇u)
        #     = E : (symm ∇) u + η : d/dt (symm ∇) u
        #     = E : (symm ∇) u + η : (symm ∇) du/dt
        #     = E : (symm ∇) u + η : (symm ∇) V
        #     = E : (symm ∇) u + τ E : (symm ∇) V
        #     =: ℒ(u) + τ ℒ(V)   [constitutive law]
        #
        #   V = ∂u/∂t + (a·∇) u   [velocity variable]
        #
        # where we have defined the constitutive linear operator
        #
        #   ℒ(...) = E : (symm ∇) (...)
        #
        # Note that now:
        #   - The constitutive equation does not explicitly mention the axial
        #     motion (no `a·∇` or `V·∇`); this is absorbed by the `V` on the RHS.
        #   - The axial motion is incorporated in an extremely simple fashion
        #     by the equation connecting `V` and `u`.
        #
        # ------------------------------------------------------------
        #
        # - Valid boundary conditions:
        #   - Displacement boundary: `v` given, no condition on `σ`
        #     - Dirichlet boundary for `v`; those rows of `F_v` removed, ∫ ds terms don't matter.
        #     - Affects `σ` automatically, via the step 1 and step 2 updates.
        #   - Stress (traction) boundary: `n·σ` given, no condition on `v`
        #     - Dirichlet boundary for `σ`; those rows of `F_σ` removed.
        #     - Need to include the -∫ [n·σ]·ψ ds term in `F_v`.
        #
        # θ integration:
        dvdt = (v - v_n) / dt
        U = (1 - θ) * u_n + θ * u_  # known
        V = (1 - θ) * v_n + θ * v   # unknown!
        Σ = (1 - θ) * σ_n + θ * σ_  # known
        # # Backward Euler integration:
        # U = u_
        # V = v
        # Σ = σ_
        F_v = (ρ * dot(dvdt, ψ) * dx +
               2 * ρ * advw(a, V, ψ, n) -
               ρ * dot(dot(a, nabla_grad(U)), dot(a, nabla_grad(ψ))) * dx +  # from +∫ ρ [(a·∇)(a·∇)u]·ψ dx
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(U)), ψ)) * ds +
               inner(Σ.T, ε(ψ)) * dx -
               dot(dot(n, Σ.T), ψ) * ds -
               ρ * dot(b, ψ) * dx)

        # SUPG: streamline upwinding Petrov-Galerkin.
        τ_SUPG = (α0 / self.V.ufl_element().degree()) * (1 / (θ * dt) + 2 * mag(a) / he + 4 * mag(a)**2 / he**2)**-1  # [τ] = s  # TODO: tune value
        # The residual is evaluated elementwise in strong form, at the end of the timestep.
        R = (ρ * ((v - v_n) / dt + 2 * advs(a, v) + advs(a, advs(a, u_))) -
             div(σ_) - ρ * b)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        F_v += F_SUPG

        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)
        # self.a_u = lhs(F_u)
        # self.L_u = rhs(F_u)
        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)
        self.a_v = lhs(F_v)
        self.L_v = rhs(F_v)

    def step(self) -> typing.Tuple[int, int, int, typing.Tuple[int, float]]:
        """Take a timestep of length `self.dt`.

        Updates the latest computed solution.
        """
        # # # Set up manual patch-averaging
        # # if not hasattr(self, "stash_initialized"):
        # #     from ..meshfunction import cellvolume
        # #     self.VtoVdG0, self.VdG0tocell = meshmagic.map_dG0(self.V, self.VdG0)
        # #     self.cell_volume_VdG0 = cellvolume(self.VdG0.mesh())
        # #     self.QtoQdG0, self.QdG0tocell = meshmagic.map_dG0(self.Q, self.QdG0)
        # #     self.cell_volume_QdG0 = cellvolume(self.QdG0.mesh())
        # #     self.stash_initialized = True
        # def postprocessV(u: Function):
        #     # `dolfin.interpolate` doesn't support quads, so we can either patch-average manually,
        #     # or use `dolfin.project` both ways.
        #     # u.assign(project(interpolate(u, self.VdG0), self.V))
        #     u.assign(project(project(u, self.VdG0), self.V))
        #     # u.assign(meshmagic.patch_average(u, self.VdG0, self.VtoVdG0, self.VdG0tocell, self.cell_volume_VdG0))
        # def postprocessQ(q: Function):
        #     # q.assign(project(interpolate(q, self.QdG0), self.Q))
        #     q.assign(project(project(q, self.QdG0), self.Q))
        #     # q.assign(meshmagic.patch_average(q, self.QdG0, self.QtoQdG0, self.QdG0tocell, self.cell_volume_QdG0))

        # `dolfin.errornorm` doesn't support quad elements, because it uses `dolfin.interpolate`.
        # Do the same thing, but avoid interpolation.
        #
        # Note this implies we cannot then use a higher-degree dG space to compute the norm,
        # like `dolfin.errornorm` does, hence this won't be as accurate. But maybe it's enough
        # for basic convergence monitoring.
        def errnorm(u, u_prev, norm_type):
            e = Function(self.V)
            e.assign(u)
            e.vector().axpy(-1.0, u_prev.vector())
            return norm(e, norm_type=norm_type, mesh=self.mesh)

        begin("Solve timestep")

        v_prev = Function(self.V)
        it1s = []
        it2s = []
        it3s = []
        for _ in range(self.maxit):
            v_prev.assign(self.v_)  # convergence monitoring

            # # NOTE: Old code, kept for documentation purposes; we no longer have `self.bcu`,
            # # as the newer mass-lumped algorithm does not need it.
            # #
            # # Step 1: update `u`
            # A1 = assemble(self.a_u)
            # b1 = assemble(self.L_u)
            # [bc.apply(A1) for bc in self.bcu]
            # [bc.apply(b1) for bc in self.bcu]
            # if not self.bcu:
            #     A1_PETSc = as_backend_type(A1)
            #     A1_PETSc.set_near_nullspace(self.null_space)
            #     A1_PETSc.set_nullspace(self.null_space)
            #     self.null_space.orthogonalize(b1)
            # it1s.append(solve(A1, self.u_.vector(), b1, 'bicgstab', 'sor'))

            # Mass-lumped direct algebraic update.
            #
            # This update method ignores the spatial connections between the DOFs.
            # Seems numerically more stable for the Kelvin-Voigt material than the
            # consistent-mass FEM update; the consistent update tends to produce
            # checkerboard oscillations at least when `u` and `v` use the same basis.
            #
            # We have
            #   ∂u/∂t = v
            # which discretizes into
            #   (u_ - u_n) / Δt = v
            #   u_ - u_n = Δt v
            #   u_ = u_n + Δt v
            # Here
            #   v = (1 - θ) v_n + θ v_
            # is the approximation of `v` consistent with the θ integration scheme.
            θ = self.θ
            dt = self.dt
            # This is what we want to do:
            #   V = (1 - θ) * self.v_n.vector()[:] + θ * self.v_.vector()[:]
            #   self.u_.vector()[:] = self.u_n.vector()[:] + dt * V
            # Maybe faster to update in-place using the low-level PETSc vector API?
            self.u_.assign(self.u_n)
            self.u_.vector().axpy(dt * (1 - θ), self.v_n.vector())
            self.u_.vector().axpy(dt * θ, self.v_.vector())
            it1s.append(1)

            # # Postprocess `u` to eliminate numerical oscillations (no need if function spaces are ok)
            # postprocessV(self.u_)

            # Step 2: update `σ`
            #
            # For a linear elastic material, the equation for the stress is symmetric.
            # For linear viscoelastic models, treating the a·∇ε term produces asymmetry.
            #
            # So let's use bicgstab.
            #
            A2a = assemble(self.a_εu)
            b2a = assemble(self.L_εu)
            solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
            A2b = assemble(self.a_εv)
            b2b = assemble(self.L_εv)
            solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

            A2 = assemble(self.a_σ)
            b2 = assemble(self.L_σ)
            [bc.apply(A2) for bc in self.bcσ]
            [bc.apply(b2) for bc in self.bcσ]
            it2s.append(solve(A2, self.σ_.vector(), b2, 'bicgstab', 'sor'))

            # # Postprocess `σ` to eliminate numerical oscillations (no need if function spaces are ok)
            # postprocessQ(self.σ_)

            # Step 3: tonight's main event (solve momentum equation for `v`)
            A3 = assemble(self.a_v)
            b3 = assemble(self.L_v)
            [bc.apply(A3) for bc in self.bcv]
            [bc.apply(b3) for bc in self.bcv]
            # Eliminate rigid-body motion solutions of momentum equation (for Krylov solvers)
            #
            # `set_near_nullspace`: "Attach near nullspace to matrix (used by preconditioners,
            #                        such as smoothed aggregation algebraic multigrid)"
            # `set_nullspace`:      "Attach nullspace to matrix (typically used by Krylov solvers
            #                        when solving singular systems)"
            #
            # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d4/db0/classdolfin_1_1PETScMatrix.html#aeb0152c4382d473ae6a93841f721260c
            #
            if not self.bcv:
                A3_PETSc = as_backend_type(A3)
                A3_PETSc.set_near_nullspace(self.null_space)
                A3_PETSc.set_nullspace(self.null_space)
                # TODO: What goes wrong here? Is it that the null space of the other linear models
                # is subtly different from the null space of the linear elastic model? So telling
                # the preconditioner to "watch out for rigid-body modes" is fine, but orthogonalizing
                # the load function against the wrong null space corrupts the loading?
                self.null_space.orthogonalize(b3)
            it3s.append(solve(A3, self.v_.vector(), b3, 'bicgstab', 'hypre_amg'))

            # # Postprocess `v` to eliminate numerical oscillations (no need if function spaces are ok)
            # postprocessV(self.v_)

            # e = errornorm(self.v_, v_prev, 'h1', 0, self.mesh)  # u, u_h, kind, degree_rise, optional_mesh
            e = errnorm(self.v_, v_prev, "h1")
            if e < self.tol:
                break

            # # relaxation / over-relaxation to help system iteration converge - does not seem to help here
            # import dolfin
            # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
            #     print(f"After iteration {(_ + 1)}: ‖v - v_prev‖_H1 = {e}")
            # if e < 1e-3:
            #     γ = 1.05
            #     self.v_.vector()[:] = (1 - γ) * v_prev.vector()[:] + γ * self.v_.vector()[:]

        # # DEBUG: do we have enough boundary conditions in the discrete system?
        # import numpy as np
        # print(np.linalg.matrix_rank(A.array()), np.linalg.norm(A.array()))
        # print(sum(np.array(b) != 0.0), np.linalg.norm(np.array(b)), np.array(b))

        end()

        it1 = sum(it1s)
        it2 = sum(it2s)
        it3 = sum(it3s)
        return it1, it2, it3, ((_ + 1), e)

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This makes the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.
        """
        self.u_n.assign(self.u_)
        self.v_n.assign(self.v_)
        self.σ_n.assign(self.σ_)

# --------------------------------------------------------------------------------

# TODO: This steady-state solver does not work yet. See `SteadyStateEulerianSolidPrimal`, which works.
class SteadyStateEulerianSolid:
    """Axially moving linear solid, small-displacement Eulerian formulation.

    Like `EulerianSolid`, but steady state.

    NOTE: WORK IN PROGRESS, does not work correctly yet.
    See `SteadyStateEulerianSolidPrimal`, which works.

    Diriclet BCs are now given for `u` (NOTE!) and `σ`.

    Note `v = ∂u/∂t ≡ 0`, because we are in an Eulerian steady state.
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC]):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Monolithic formulation.
        #
        # Using a `MixedFunctionSpace` fails for some reason. Instead, the way to
        # do this is to set up a `MixedElement` and a garden-variety `FunctionSpace`
        # on that, and then split as needed. Then set Dirichlet BCs on the appropriate
        # `S.sub(j)` (those may also have their own second-level `.sub(k)` if they are
        # vector/tensor fields).
        #
        e = MixedElement(V.ufl_element(), Q.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, σ = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        ψ, φ = TestFunctions(S)
        s_ = Function(S)
        u_, σ_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, σ_ = s_.sub(0), s_.sub(1)  # if you want the `Function` (for plotting etc.)

        self.V = V
        self.Q = Q
        self.VdG0 = VectorFunctionSpace(self.mesh, "DG", 0)
        self.QdG0 = TensorFunctionSpace(self.mesh, "DG", 0)

        # This algorithm uses `P` only for strain visualization.
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.u, self.σ = u, σ  # trials
        self.ψ, self.φ = ψ, φ  # tests
        self.u_, self.σ_ = u_, σ_  # solution

        self.S = S
        self.s_ = s_

        # Unused, all zeros, but provided for API reasons (all solvers supply a plottable velocity field;
        # for this one, the velocity is zero).
        self.v_ = Function(V)

        # Set up the null space for removal in the Krylov solver.
        fus = null_space_fields(self.mesh.geometric_dimension())
        # In a mixed formulation, we must insert zero functions for the other fields:
        zeroV = Function(V)
        zeroV.vector()[:] = 0.0
        zeroQ = Function(Q)
        zeroQ.vector()[:] = 0.0
        # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
        assigner = FunctionAssigner(S, [V, Q])  # receiving space, assigning space
        fss = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fss, fus):
            assigner.assign(fs, [project(fu, V), zeroQ])
        null_space_basis = [fs.vector() for fs in fss]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        # PDE system iteration parameters.
        # User-configurable (`solver.maxit = ...`), but not a major advertised feature.
        self.maxit = 100  # maximum number of system iterations per timestep
        self.tol = 1e-8  # system iteration tolerance, ‖v - v_prev‖_H1 (over the whole domain)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u = self.u      # unknown
        ψ = self.ψ      # test

        # Stress
        σ = self.σ
        φ = self.φ

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # We build one monolithic equation.

        # Constitutive equation
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.

        Id = Identity(ε(u).geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

        # Choose constitutive model
        if self.τ == 0.0:  # Linear elastic (LE)
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u)), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            # TODO: This equation may need SUPG when τ ≠ 0, a ≠ 0.
            F_σ = (inner(σ, φ) * dx -
                   (inner(K_inner(ε(u)), sym(φ)) * dx +  # linear elastic
                    τ * dot(a, n) * inner(K_inner(ε(u)), sym(φ)) * ds -  # generated by ∫ (φ:Kη):(a·∇)ε dx
                    τ * inner(K_inner(ε(u)), advs(a, sym(φ))) * dx))  # generated by ∫ (φ:Kη):(a·∇)ε dx

        # Linear momentum balance
        #
        # This equation has no `u` when `a = 0`, so we use a monolithic formulation
        # (both equations solved as one system); this allows the solver to see the
        # dependence `σ = σ(u)`.
        F_u = (-ρ * dot(dot(a, nabla_grad(u)), dot(a, nabla_grad(ψ))) * dx +  # from +∫ ρ [(a·∇)(a·∇)u]·ψ dx
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(u)), ψ)) * ds +
               inner(σ.T, ε(ψ)) * dx -
               dot(dot(n, σ.T), ψ) * ds -
               ρ * dot(b, ψ) * dx)

        F = F_σ + F_u
        self.a = lhs(F)
        self.L = rhs(F)

        # Strains, for visualization only.
        εu = self.εu  # unknown
        εv = self.εv
        q = self.q
        F_εu = inner(εu, q) * dx - inner(ε(self.u_), sym(q)) * dx
        F_εv = inner(εv, q) * dx - inner(ε(self.v_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

    def solve(self) -> typing.Tuple[int, int, typing.Tuple[int, float]]:
        """Solve the steady state.

        The solution becomes available in `self.s_`.
        """
        begin("Solve steady state")
        A = assemble(self.a)
        b = assemble(self.L)
        [bc.apply(A) for bc in self.bcσ]
        [bc.apply(b) for bc in self.bcσ]
        [bc.apply(A) for bc in self.bcu]
        [bc.apply(b) for bc in self.bcu]
        if not self.bcu:
            A_PETSc = as_backend_type(A)
            A_PETSc.set_near_nullspace(self.null_space)
            A_PETSc.set_nullspace(self.null_space)
            self.null_space.orthogonalize(b)
        it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')
        end()

        # VISUALIZATION PURPOSES ONLY
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

        return it

# --------------------------------------------------------------------------------

class EulerianSolidAlternative:
    """Like `EulerianSolid`, but with different splitting for the equations.

    In this formulation, the velocity variable `v` is the material parcel
    velocity in the co-moving frame (instead of just the Eulerian rate of
    `u` as in `EulerianSolid`).

    As in `EulerianSolid`, the primary variables are `v` and `σ`. In general,
    boundary conditions must be specified for either `v` or `n·σ` on each
    boundary.

    If `V0 ≠ 0`, additionally `u` needs boundary conditions on the inflow part
    of the boundary (i.e. on the part for which `a·n < 0`, where `a := (V0, 0)`).


    **Equations**:

    The formulation used by `EulerianSolidAlternative` is based on an alternative
    choice for the velocity-like variable. Let us denote the material parcel velocity,
    as measured against the co-moving frame, by `V`:

        V := du/dt

    Consider the momentum balance for a continuum:

        ρ dv/dt - ∇·σ = ρ b

    where `v` is the material parcel velocity in an inertial frame of our
    choice (the law is postulated to be Galilean invariant).

    Let us choose the laboratory frame. The material parcel velocity with
    respect to the laboratory is

        v := a + V    (*)

    where `a` is the axial drive velocity. The Eulerian description of the
    momentum balance is

        ρ ∂v/∂t + ρ (v·∇) v - ∇·σ = ρ b

    Inserting (*), we have

        ρ ∂(a + V)/∂t + ρ ((a + V)·∇) (a + V) - ∇·σ = ρ b

    which simplifies to

        ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b

    where we have used the fact that `a` is constant, and dropped the
    second-order small term `ρ (V·∇) V` (since in the small-displacement
    regime, `V` is taken to be small).

    For Kelvin-Voigt (linear elastic as special case τ = 0):

        σ = E : ε + η : dε/dt
          = E : (symm ∇u) + η : d/dt (symm ∇u)
          = E : (symm ∇) u + η : d/dt (symm ∇) u
          = E : (symm ∇) u + η : (symm ∇) du/dt
          = E : (symm ∇) u + η : (symm ∇) V
          = E : (symm ∇) u + τ E : (symm ∇) V
          =: ℒ(u) + τ ℒ(V)

    Although the constitutive law has the same form as the standard Lagrangean
    one, it is actually Eulerian; both `u` and `V` are represented using
    Eulerian fields, and also the ∇ is taken in spatial coordinates.

    In the constitutive law, we can use our Eulerian `u` and `V`, although they
    have the uniform axial motion abstracted out, because the uniform motion
    causes no strain.

    This formulation resembles Navier-Stokes, but for solids. The displacement,
    needed in the constitutive law, is obtained by considering our definition
    of `V`:

        V ≡ du/dt
          = ∂u/∂t + ((a + V)·∇) u    [based on laboratory-frame Eulerian `u`]
          ≈ ∂u/∂t + (a·∇) u          [drop 2nd order small term]

    so up to first order in the small quantities, the displacement can be
    updated by solving the linear first-order transport PDE

        ∂u/∂t + (a·∇) u = V

    The displacement `u` is an Eulerian field, measured with respect to a
    reference state where the material travels at constant axial velocity.
    Both `u` and `V` are parameterized by the laboratory coordinate `x`.
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcv: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        u = TrialFunction(V)  # no suffix: UFL symbol for unknown quantity
        w = TestFunction(V)
        v = TrialFunction(V)
        ψ = TestFunction(V)
        σ = TrialFunction(Q)
        φ = TestFunction(Q)

        u_ = Function(V)  # suffix _: latest computed approximation
        u_n = Function(V)  # suffix _n: old value (end of previous timestep)
        v_ = Function(V)
        v_n = Function(V)
        σ_ = Function(Q)
        σ_n = Function(Q)

        self.V = V
        self.Q = Q

        # This algorithm uses `P` only for strain visualization.
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.u, self.v, self.σ = u, v, σ  # trials
        self.w, self.ψ, self.φ = w, ψ, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # latest computed approximation
        self.u_n, self.v_n, self.σ_n = u_n, v_n, σ_n  # old value (end of previous timestep)

        # Set up the null space for removal in the Krylov solver.
        fus = null_space_fields(self.mesh.geometric_dimension())
        null_space_basis = [interpolate(fu, V).vector() for fu in fus]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcv = bcv
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.V0 = V0
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        # PDE system iteration parameters.
        # User-configurable (`solver.maxit = ...`), but not a major advertised feature.
        self.maxit = 100  # maximum number of system iterations per timestep
        self.tol = 1e-8  # system iteration tolerance, ‖v - v_prev‖_H1 (over the whole domain)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
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
        v = self.v
        ψ = self.ψ
        v_ = self.v_
        v_n = self.v_n

        # Stress
        σ = self.σ
        φ = self.φ
        σ_ = self.σ_
        σ_n = self.σ_n

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # The strong form of the equations we are discretizing is:
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
        #   ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b   [linear momentum balance]

        # Step 1: ∂u/∂t + (a·∇)u = v  ->  obtain `u`
        #
        dudt = (u - u_n) / dt
        U = (1 - θ) * u_n + θ * u   # unknown!
        V = (1 - θ) * v_n + θ * v_  # known ("new" value = latest iterate)
        F_u = (dot(dudt, w) * dx +
               advw(a, U, w, n) -
               dot(V, w) * dx)

        # SUPG: streamline upwinding Petrov-Galerkin. The residual is evaluated elementwise in strong form.
        deg = Constant(self.V.ufl_element().degree())
        τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he)**-1  # [τ] = s
        # R = ((u - u_n) / dt + advs(a, u) - v_)  # end of timestep
        R = ((u - u_n) / dt + advs(a, U) - V)  # consistent (θ-point in time)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, w), R) * dx
        F_u += F_SUPG

        # Step 2: solve `σ`, using `u` from step 1 (and the latest available `v`)
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.

        # θ integration:
        U = (1 - θ) * u_n + θ * u_  # known
        V = (1 - θ) * v_n + θ * v_  # known
        Σ = (1 - θ) * σ_n + θ * σ   # unknown!

        εu_ = ε(U)
        εv_ = ε(V)
        Id = Identity(εu_.geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

        # Choose constitutive model
        #
        # For Kelvin-Voigt:
        #
        #   σ = E : ε + η : dε/dt
        #     = E : (symm ∇u) + η : d/dt (symm ∇u)
        #     = E : (symm ∇) u + η : d/dt (symm ∇) u
        #     = E : (symm ∇) u + η : (symm ∇) du/dt
        #     = E : (symm ∇) u + η : (symm ∇) V
        #     = E : (symm ∇) u + τ E : (symm ∇) V
        #     =: ℒ(u) + τ ℒ(V)   [constitutive law]
        #
        if self.τ == 0.0:  # Linear elastic (LE)
            F_σ = (inner(Σ, φ) * dx -
                   inner(K_inner(εu_), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            # No transport term, because `v` already contains the transport effects.
            F_σ = (inner(Σ, φ) * dx -
                   inner(K_inner(εu_) + τ * K_inner(εv_), sym(φ)) * dx)

        # Step 3: solve `v` from momentum equation
        #
        #   ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b
        #
        # θ integration:
        dvdt = (v - v_n) / dt
        U = (1 - θ) * u_n + θ * u_  # known
        V = (1 - θ) * v_n + θ * v   # unknown!
        Σ = (1 - θ) * σ_n + θ * σ_  # known
        F_v = (ρ * dot(dvdt, ψ) * dx + ρ * advw(a, V, ψ, n) +
               inner(Σ.T, ε(ψ)) * dx - dot(dot(n, Σ.T), ψ) * ds -
               ρ * dot(b, ψ) * dx)

        # SUPG: streamline upwinding Petrov-Galerkin. The residual is evaluated elementwise in strong form.
        deg = Constant(self.V.ufl_element().degree())
        # # Very basic scaling; the resulting τ_SUPG is perhaps too large
        # # (excessive diffusion along streamlines of `a`).
        # τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he + 4 * mag(a)**2 / he**2)**-1  # [τ] = s
        #
        # Navier-Stokes uses 4 * (μ / ρ) / he² in the second-order part.
        # Since we have both elastic and viscous effects, with both shear
        # and volumetric contributions, take the largest one of these as
        # the representative second-order coefficient.
        moo = Constant(max(self.λ, 2 * self.μ, self.τ * self.λ, self.τ * 2 * self.μ))
        τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (moo / ρ) / he**2)**-1  # [τ] = s
        # R = (ρ * ((v - v_n) / dt + advs(a, v)) - div(σ_) - ρ * b)  # end of timestep
        R = (ρ * ((v - v_n) / dt + advs(a, V)) - div(Σ) - ρ * b)  # consistent (θ-point in time)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        F_v += F_SUPG

        self.a_u = lhs(F_u)
        self.L_u = rhs(F_u)
        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)
        self.a_v = lhs(F_v)
        self.L_v = rhs(F_v)

        # Strains at end of timestep, for visualization only.
        εu = self.εu  # unknown
        εv = self.εv
        q = self.q
        F_εu = inner(εu, q) * dx - inner(ε(u_), sym(q)) * dx
        F_εv = inner(εv, q) * dx - inner(ε(v_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

    def step(self) -> typing.Tuple[int, int, int, typing.Tuple[int, float]]:
        """Take a timestep of length `self.dt`.

        Updates the latest computed solution.
        """
        def errnorm(u, u_prev, norm_type):
            e = Function(self.V)
            e.assign(u)
            e.vector().axpy(-1.0, u_prev.vector())
            return norm(e, norm_type=norm_type, mesh=self.mesh)

        begin("Solve timestep")

        v_prev = Function(self.V)
        it1s = []
        it2s = []
        it3s = []
        for sysit in range(self.maxit):
            v_prev.assign(self.v_)  # convergence monitoring

            # Step 1: update `u`
            A1 = assemble(self.a_u)
            b1 = assemble(self.L_u)
            # Inflow boundary conditions: only needed in axially moving case
            if self.V0 != 0:
                [bc.apply(A1) for bc in self.bcu]
                [bc.apply(b1) for bc in self.bcu]
            it1s.append(solve(A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg'))

            # Step 2: update `σ`
            #
            # A2a = assemble(self.a_εu)
            # b2a = assemble(self.L_εu)
            # solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
            # A2b = assemble(self.a_εv)
            # b2b = assemble(self.L_εv)
            # solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

            A2 = assemble(self.a_σ)
            b2 = assemble(self.L_σ)
            [bc.apply(A2) for bc in self.bcσ]
            [bc.apply(b2) for bc in self.bcσ]
            it2s.append(solve(A2, self.σ_.vector(), b2, 'bicgstab', 'sor'))

            # Step 3: tonight's main event (solve momentum equation for `v`)
            A3 = assemble(self.a_v)
            b3 = assemble(self.L_v)
            [bc.apply(A3) for bc in self.bcv]
            [bc.apply(b3) for bc in self.bcv]
            if not self.bcv:
                A3_PETSc = as_backend_type(A3)
                A3_PETSc.set_near_nullspace(self.null_space)
                A3_PETSc.set_nullspace(self.null_space)
                # TODO: What goes wrong here? Is it that the null space of the other linear models
                # is subtly different from the null space of the linear elastic model? So telling
                # the preconditioner to "watch out for rigid-body modes" is fine, but orthogonalizing
                # the load function against the wrong null space corrupts the loading?
                self.null_space.orthogonalize(b3)
            it3s.append(solve(A3, self.v_.vector(), b3, 'bicgstab', 'hypre_amg'))

            # e = errornorm(self.v_, v_prev, 'h1', 0, self.mesh)  # u, u_h, kind, degree_rise, optional_mesh
            e = errnorm(self.v_, v_prev, "h1")
            if e < self.tol:
                break

            # # relaxation / over-relaxation to help system iteration converge - does not seem to help here
            # import dolfin
            # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
            #     print(f"After iteration {(_ + 1)}: ‖v - v_prev‖_H1 = {e}")
            # if e < 1e-3:
            #     γ = 1.05
            #     self.v_.vector()[:] = (1 - γ) * v_prev.vector()[:] + γ * self.v_.vector()[:]

        # # DEBUG: do we have enough boundary conditions in the discrete system?
        # import numpy as np
        # print(np.linalg.matrix_rank(A.array()), np.linalg.norm(A.array()))
        # print(sum(np.array(b) != 0.0), np.linalg.norm(np.array(b)), np.array(b))

        end()

        it1 = sum(it1s)
        it2 = sum(it2s)
        it3 = sum(it3s)
        return it1, it2, it3, (1 + sysit, e)  # e = final error

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This makes the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.

        This also computes the strain fields (available for visualization
        purposes).
        """
        # VISUALIZATION PURPOSES ONLY
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

        self.u_n.assign(self.u_)
        self.v_n.assign(self.v_)
        self.σ_n.assign(self.σ_)

# --------------------------------------------------------------------------------

# NOTE: This algorithm does not work yet.
class SteadyStateEulerianSolidAlternative:
    """Axially moving linear solid, small-displacement Eulerian formulation.

    Like `EulerianSolidAlternative`, but steady state.

    Note `v = du/dt = (a·∇) u`, because we are in an Eulerian steady state.

    NOTE: The equation system is monolithic, so no system iterations are needed.
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcv: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC]):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Monolithic formulation.
        #
        # Using a `MixedFunctionSpace` fails for some reason. Instead, the way to
        # do this is to set up a `MixedElement` and a garden-variety `FunctionSpace`
        # on that, and then split as needed. Then set Dirichlet BCs on the appropriate
        # `S.sub(j)` (those may also have their own second-level `.sub(k)` if they are
        # vector/tensor fields).
        #
        e = MixedElement(V.ufl_element(), V.ufl_element(), Q.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v, σ = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        w, ψ, φ = TestFunctions(S)
        s_ = Function(S)
        u_, v_, σ_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, v_, σ_ = s_.sub(0), s_.sub(1), s_.sub(2)  # if you want the `Function` (for plotting etc.)

        self.V = V
        self.Q = Q

        # This algorithm uses `P` only for strain visualization.
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.u, self.v, self.σ = u, v, σ  # trials
        self.w, self.ψ, self.φ = w, ψ, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # solution

        self.S = S
        self.s_ = s_

        # Set up the null space for removal in the Krylov solver.
        fus = null_space_fields(self.mesh.geometric_dimension())
        # In a mixed formulation, we must insert zero functions for the other fields:
        zeroV = Function(V)
        zeroV.vector()[:] = 0.0
        zeroQ = Function(Q)
        zeroQ.vector()[:] = 0.0
        # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
        assigner = FunctionAssigner(S, [V, V, Q])  # receiving space, assigning space
        fssu = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssu, fus):
            assigner.assign(fs, [project(fu, V), zeroV, zeroQ])
        fssv = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssv, fus):
            assigner.assign(fs, [zeroV, project(fu, V), zeroQ])
        null_space_basis = [fs.vector() for fs in fssu + fssv]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcv = bcv
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.V0 = V0
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        # PDE system iteration parameters.
        # User-configurable (`solver.maxit = ...`), but not a major advertised feature.
        self.maxit = 100  # maximum number of system iterations per timestep
        self.tol = 1e-8  # system iteration tolerance, ‖v - v_prev‖_H1 (over the whole domain)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u = self.u      # unknown
        w = self.w      # test

        # Material parcel velocity in co-moving frame,  v = (a·∇)u  (steady state)
        v = self.v
        ψ = self.ψ

        # Stress
        σ = self.σ
        φ = self.φ

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # We build one monolithic equation.

        # Constitutive equation
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.

        Id = Identity(ε(u).geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`

        # Choose constitutive model
        if self.τ == 0.0:  # Linear elastic (LE)
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u)), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u)) + τ * K_inner(ε(v)), sym(φ)) * dx)

        F_u = (ρ * advw(a, v, w, n) +
               inner(σ.T, ε(w)) * dx - dot(dot(n, σ.T), w) * ds -
               ρ * dot(b, w) * dx)
        F_v = (dot(v, ψ) * dx - advw(a, u, ψ, n))

        # # SUPG: streamline upwinding Petrov-Galerkin. The residual is evaluated elementwise in strong form.
        # deg = Constant(self.V.ufl_element().degree())
        # moo = Constant(max(self.λ, 2 * self.μ, self.τ * self.λ, self.τ * 2 * self.μ))
        # τ_SUPG = (α0 / deg) * (2 * mag(a) / he + 4 * (moo / ρ) / he**2)**-1  # [τ] = s
        # R = (ρ * advs(a, v) - div(σ) - ρ * b)
        # F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, w), R) * dx
        # F_u += F_SUPG
        #
        # deg = Constant(self.V.ufl_element().degree())
        # τ_SUPG = (α0 / deg) * (2 * mag(a) / he)**-1  # [τ] = s
        # R = (v - advs(a, u))
        # F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        # F_v += F_SUPG

        F = F_u + F_v + F_σ
        self.a = lhs(F)
        self.L = rhs(F)

        # Strains, for visualization only.
        εu = self.εu  # unknown
        εv = self.εv
        q = self.q
        F_εu = inner(εu, q) * dx - inner(ε(self.u_), sym(q)) * dx
        F_εv = inner(εv, q) * dx - inner(ε(self.v_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

    def solve(self) -> typing.Tuple[int, int, typing.Tuple[int, float]]:
        """Solve the steady state.

        The solution becomes available in `self.s_`.
        """
        begin("Solve steady state")
        A = assemble(self.a)
        b = assemble(self.L)
        [bc.apply(A) for bc in self.bcu]
        [bc.apply(b) for bc in self.bcu]
        [bc.apply(A) for bc in self.bcv]
        [bc.apply(b) for bc in self.bcv]
        [bc.apply(A) for bc in self.bcσ]
        [bc.apply(b) for bc in self.bcσ]
        if not self.bcu:
            A_PETSc = as_backend_type(A)
            A_PETSc.set_near_nullspace(self.null_space)
            A_PETSc.set_nullspace(self.null_space)
            self.null_space.orthogonalize(b)
        it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')

        # VISUALIZATION PURPOSES ONLY
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

        end()
        return it

# --------------------------------------------------------------------------------

class EulerianSolidPrimal:
    """Like `EulerianSolidAlternative`, but using only `u` and `v`.

    Here the linear momentum balance law is used for determining `u`.

    `v` is determined by L2 projection, using the definition

      v := du/dt = ∂u/∂t + (a·∇)u

    The projection is performed using the *unknown* `u`, by including this equation
    into a monolithic equation system.

    Boundary conditions should be set on `u`; `v` takes no BCs.

    Boundary stresses are enforced using a Neumann BC. `bcσ` is a single expression
    that will be evaluated at boundaries that do not have a boundary condition for
    `u`.

    Stresses are computed at timestep commit, and provided for visualization only.

    NOTE: This solver uses MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
    instead of Krylov methods to solve the linear equation system, so the returned
    iteration counts are just meaningless placeholders to provide a unified API.

    NOTE: The equation system is monolithic, so no system iterations are needed.
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcv: typing.List[DirichletBC],
                 bcσ: Expression,
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        e = MixedElement(V.ufl_element(), V.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        w, ψ = TestFunctions(S)
        s_ = Function(S)  # suffix _: latest computed approximation
        u_, v_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, v_ = s_.sub(0), s_.sub(1)  # if you want the `Function` (for plotting etc.)
        s_n = Function(S)  # suffix _n: old value (end of previous timestep)
        u_n, v_n = split(s_n)

        # For separate equation, for stress visualization
        σ = TrialFunction(Q)
        φ = TestFunction(Q)
        σ_ = Function(Q)

        self.V = V
        self.Q = Q

        # This algorithm uses `P` only for strain visualization.
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.u, self.v, self.σ = u, v, σ  # trials
        self.w, self.ψ, self.φ = w, ψ, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # latest computed approximation
        self.u_n, self.v_n = u_n, v_n  # old value (end of previous timestep)

        self.S = S
        self.s_ = s_
        self.s_n = s_n

        # Set up the null space for removal in the Krylov solver. (TODO: currently unused in this solver)
        fus = null_space_fields(self.mesh.geometric_dimension())
        # In a mixed formulation, we must insert zero functions for the other fields:
        zeroV = Function(V)
        zeroV.vector()[:] = 0.0
        # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
        assigner = FunctionAssigner(S, [V, V])  # receiving space, assigning space
        fssu = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssu, fus):
            assigner.assign(fs, [project(fu, V), zeroV])
        fssv = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssv, fus):
            assigner.assign(fs, [zeroV, project(fu, V)])
        null_space_basis = [fs.vector() for fs in fssu + fssv]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcv = bcv

        # Neumann BC for stress
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.V0 = V0
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        self.tol = 1e-8  # TODO: unused, but the main script expects to have it because other solvers here do.

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
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
        v = self.v
        ψ = self.ψ
        v_ = self.v_
        v_n = self.v_n

        # Stress
        σ = self.σ
        φ = self.φ

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # The strong form of the equations we are discretizing is:
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
        #   ρ ∂V/∂t + ρ (a·∇) V - ∇·σ = ρ b   [linear momentum balance]

        # Monolithic equation system.
        U = (1 - θ) * u_n + θ * u
        V = (1 - θ) * v_n + θ * v
        dudt = (u - u_n) / dt
        dvdt = (v - v_n) / dt
        Σ0 = self.bcσ  # Neumann BC for stress

        # `σ`, using unknown `u` and `v`
        #
        # TODO: Extend the constitutive model. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.
        #
        # TODO: Add Neumann BC for `(n·∇)u`; we need a zero normal gradient
        # TODO: BC in the analysis of 3D printing. For details,
        # TODO: see comments in `SteadyStateEulerianSolidPrimal`.

        # Constitutive model
        Id = Identity(ε(u).geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`
        if self.τ == 0.0:  # Linear elastic (LE)
            # for equation
            Σ = K_inner(ε(U))
            # for visualization
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u_)), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            Σ = K_inner(ε(U)) + τ * K_inner(ε(V))
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u_)) + τ * K_inner(ε(v_)), sym(φ)) * dx)

        F_u = (ρ * (dot(dvdt, w) * dx + advw(a, V, w, n)) +
               inner(Σ.T, ε(w)) * dx - dot(dot(n, Σ0.T), w) * ds -
               ρ * dot(b, w) * dx)
        F_v = (dot(V, ψ) * dx -
               (dot(dudt, ψ) * dx + advw(a, U, ψ, n)))

        # SUPG (doesn't seem to actually do much here)
        deg = Constant(self.V.ufl_element().degree())
        moo = Constant(max(self.λ, 2 * self.μ, self.τ * self.λ, self.τ * 2 * self.μ))
        τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (moo / ρ) / he**2)**-1  # [τ] = s
        R = (ρ * (dvdt + advs(a, V)) - div(Σ) - ρ * b)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, w), R) * dx
        F_u += F_SUPG

        deg = Constant(self.V.ufl_element().degree())
        τ_SUPG = (α0 / deg) * (1 / (θ * dt) + 2 * mag(a) / he)**-1  # [τ] = s
        R = (V - (dudt + advs(a, U)))
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        F_v += F_SUPG

        F = F_u + F_v
        self.a = lhs(F)
        self.L = rhs(F)

        # For visualization only.
        εu = self.εu  # unknown
        εv = self.εv
        q = self.q
        F_εu = inner(εu, q) * dx - inner(ε(u_), sym(q)) * dx
        F_εv = inner(εv, q) * dx - inner(ε(v_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)

    def step(self) -> typing.Tuple[int, int, int, typing.Tuple[int, float]]:
        """Take a timestep of length `self.dt`.

        Updates the latest computed solution.
        """
        begin("Solve timestep")

        A = assemble(self.a)
        b = assemble(self.L)
        [bc.apply(A) for bc in self.bcu]
        [bc.apply(b) for bc in self.bcu]
        [bc.apply(A) for bc in self.bcv]
        [bc.apply(b) for bc in self.bcv]
        # TODO: The rigid-body mode remover is useless if we use a direct method (only applies to Krylov).
        if not self.bcu:
            A3_PETSc = as_backend_type(A)
            A3_PETSc.set_near_nullspace(self.null_space)
            A3_PETSc.set_nullspace(self.null_space)
            # TODO: What goes wrong here? Is it that the null space of the other linear models
            # is subtly different from the null space of the linear elastic model? So telling
            # the preconditioner to "watch out for rigid-body modes" is fine, but orthogonalizing
            # the load function against the wrong null space corrupts the loading?
            self.null_space.orthogonalize(b)
        # For some reason, Krylov methods don't work well here. MUMPS seems to get the correct solution.
        # it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')
        solve(A, self.s_.vector(), b, 'mumps')
        it = 1

        end()

        return 0, 0, it, (1, 0.0)

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This makes the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.

        This also computes the strain fields (available for visualization
        purposes).
        """
        # VISUALIZATION PURPOSES ONLY
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')

        A2 = assemble(self.a_σ)
        b2 = assemble(self.L_σ)
        solve(A2, self.σ_.vector(), b2, 'bicgstab', 'sor')

        self.s_n.assign(self.s_)

# --------------------------------------------------------------------------------

class SteadyStateEulerianSolidPrimal:
    """Axially moving linear solid, small-displacement Eulerian formulation.

    Like `EulerianSolidPrimal`, but steady state.

    Note `v = du/dt = (a·∇) u`, because we are in an Eulerian steady state.

    In this formulation we do not need to differentiate strains. This solver
    uses a primal formulation in terms of `u` and `v` (the constitutive law
    is inserted into the linear momentum balance).

    Boundary stresses are enforced using a Neumann BC. `bcσ` is a single expression
    that will be evaluated at boundaries that do not have a boundary condition for
    `u`.

    Stresses are computed for visualization only.

    NOTE: The equation system is monolithic, so no system iterations are needed.
    """
    def __init__(self, V: VectorFunctionSpace,
                 Q: TensorFunctionSpace,
                 P: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float, τ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcv: typing.List[DirichletBC],
                 bcσ: Expression):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Monolithic formulation.
        #
        # Using a `MixedFunctionSpace` fails for some reason. Instead, the way to
        # do this is to set up a `MixedElement` and a garden-variety `FunctionSpace`
        # on that, and then split as needed. Then set Dirichlet BCs on the appropriate
        # `S.sub(j)` (those may also have their own second-level `.sub(k)` if they are
        # vector/tensor fields).
        #
        e = MixedElement(V.ufl_element(), V.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        w, ψ = TestFunctions(S)
        s_ = Function(S)
        u_, v_ = split(s_)  # gives `ListTensor` (for UFL forms in the monolithic system), not `Function`
        # u_, v_ = s_.sub(0), s_.sub(1)  # if you want the `Function` (for plotting etc.)

        # For separate equation, for stress visualization
        σ = TrialFunction(Q)
        φ = TestFunction(Q)
        σ_ = Function(Q)

        self.V = V
        self.Q = Q

        # This algorithm uses `P` only for strain visualization.
        self.P = P
        self.q = TestFunction(P)
        self.εu = TrialFunction(P)
        self.εv = TrialFunction(P)
        self.εu_ = Function(P)
        self.εv_ = Function(P)

        self.u, self.v, self.σ = u, v, σ  # trials
        self.w, self.ψ, self.φ = w, ψ, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # solution

        self.S = S
        self.s_ = s_

        # Set up the null space for removal in the Krylov solver.
        fus = null_space_fields(self.mesh.geometric_dimension())
        # In a mixed formulation, we must insert zero functions for the other fields:
        zeroV = Function(V)
        zeroV.vector()[:] = 0.0
        # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
        assigner = FunctionAssigner(S, [V, V])  # receiving space, assigning space
        fssu = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssu, fus):
            assigner.assign(fs, [project(fu, V), zeroV])
        fssv = [Function(S) for _ in range(len(fus))]
        for fs, fu in zip(fssv, fus):
            assigner.assign(fs, [zeroV, project(fu, V)])
        null_space_basis = [fs.vector() for fs in fssu + fssv]
        basis = VectorSpaceBasis(null_space_basis)
        basis.orthonormalize()
        self.null_space = basis

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcv = bcv

        # Neumann BC expression for stress
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.V0 = V0
        self.a = Constant((V0, 0))

        # Specific body force (N / kg = m / s²). FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._τ = Constant(τ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        # PDE system iteration parameters.
        # User-configurable (`solver.maxit = ...`), but not a major advertised feature.
        self.maxit = 100  # maximum number of system iterations per timestep
        self.tol = 1e-8  # system iteration tolerance, ‖v - v_prev‖_H1 (over the whole domain)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    τ = ufl_constant_property("τ", doc="Kelvin-Voigt retardation time [s]")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u = self.u      # unknown
        u_ = self.u_    # computed solution (for visualization)
        w = self.w      # test

        # Material parcel velocity in co-moving frame,  v = (a·∇)u  (steady state)
        v = self.v
        v_ = self.v_    # computed solution (for visualization)
        ψ = self.ψ

        # Stress
        σ = self.σ
        φ = self.φ

        # Velocity field for axial motion
        a = self.a

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        τ = self._τ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # Define variational problem
        #
        # We build one monolithic equation.

        # TODO: Extend the constitutive model for `pdes.eulerian_solid`
        #  - Account for thermal strain.
        #    - Used to be eq. (768) in report, but report reorganized; relevant details
        #      summarized in the detailed comment further below.
        #    - Need a FEM field for temperature `T`, and new parameters `α` and `T0`.
        #      In general, must provide both `α` and `∂α/∂T`; use SymPy to generate?
        #    - Elastic parameters can also be temperature-dependent. We can make
        #      them into FEM fields here, and initialize to the constant value passed
        #      into the constructor. It is then up to the main script to update the
        #      values, using `T = θ * heatsolver.u_n + (1 - θ) * heatsolver.u_`
        #      (so that the time discretization matches the other terms) to compute
        #      the field values. This must be projected into the correct function
        #      space. If the spaces match, a relation that only needs the local
        #      value of `T` can be implemented as a PETSc vector update at the DOF
        #      level, which is cheap.
        #    - In dynamic case, need access also to `∂T/∂t`; thermal solver doesn't
        #      currently store it, but the value consistent with the time discretization
        #      can be obtained as `(heatsolver.u_ - heatsolver.u_n) / dt`.
        #      Maybe the right API here is to pass in the objects for `T_` and `T_n`,
        #      so these can be inserted into the UFL form.
        #  - Orthotropic Kelvin-Voigt.
        #    - Need new symmetry groups, the operator `K:(...)` changes slightly.
        #      Useful to have a conversion between Voigt notation (matrices) and
        #      symmetric rank-4 tensors to compactly specify anisotropic models.
        #    - The thermal expansion tensor `α` also changes. An orthotropic one
        #      can be constructed as a sum of three rank-2 tensors, each describing
        #      the thermal expansion behavior along one of the orthogonal material axes.
        #  - Isotropic SLS (Zener).
        #    - In the `EulerianSolidAlternative` formulation, requires solving a PDE.
        #      LHS includes a term `dσ/dt = ∂σ/∂t + (a·∇)σ`; could we use the same
        #      auxiliary variable technique as for `v`?
        #    - In `EulerianSolidPrimal`... ouch! We could probably take a page from
        #      Marynowski's studies on viscoelastic materials: differentiate the
        #      whole linear momentum balance in order to make the LHS of the
        #      constitutive equation appear in it. Then substitute as usual.
        #      This will raise the order (at least w.r.t. time) by one.
        #      Might be able to avoid raising order w.r.t. space by defining
        #      suitable auxiliary variables (at least `A := dv/dt`).
        #  - Orthotropic SLS (Zener). Apply new symmetry groups here, too.

        # TODO: Add Neumann BC for `(n·∇)u`; we need a zero normal gradient
        # TODO: BC in the analysis of 3D printing.
        #
        # In the primal formulation, we can use the same technique for making our
        # Neumann BC into zero-normal-gradient as is used for Navier-Stokes.
        #
        # For a model that includes the effects of thermal expansion,
        # the total strain `ε` (which appears in the kinematic relation)
        # consists of two parts:
        #
        #   ε = εve + εth
        #
        # where `εve` is the viscoelastic strain, and `εth` is the thermal strain.
        # Only the viscoelastic strain produces stress.
        #
        # Recall that written with the Kelvin-Voigt retardation time `τ`,
        # the stress for Kelvin-Voigt is given by
        #
        #   σ = K : εve + τ K : dεve/dt
        #
        # where, now that thermal effects are included,
        #
        #   εve = ε - εth
        #
        # The thermal strain is given by
        #
        #   εth = α [T - T0]
        #
        # Therefore, the stress becomes
        #
        #   σ = K : [ε - εth] + τ K : d[ε - εth]/dt
        #
        # Expanding the brackets, inserting the thermal strain εth, and annotating,
        #
        #   σ = K : ε + τ K dε/dt       (elastic and viscous responses)
        #     - K : α [T - T0]          (thermoelastic response)
        #     - τ K : d[α [T - T0]]/dt  (thermoviscous response)
        #
        # Performing the differentiations in the thermal terms,
        #
        #   σ = K : ε + τ K dε/dt       (elastic and viscous)
        #     - K : α [T - T0]          (thermoelastic)
        #     - τ K : α dT/dt           (thermoviscous)
        #     - τ K : dα/dt [T - T0]    (thermoviscous, with time-varying α)
        #
        # Taking `α = α(T)`,
        #
        #   σ = K : ε + τ K dε/dt           (elastic and viscous)
        #     - K : α [T - T0]              (thermoelastic)
        #     - τ K : α dT/dt               (thermoviscous)
        #     - τ K : ∂α/∂T dT/dt [T - T0]  (thermoviscous, with temperature-varying α)
        #
        # Expanding the material derivative `dα/dt` (approximately, ignoring the
        # material parcel motion with respect to the co-moving frame),
        #
        #   σ = K : ε + τ K dε/dt
        #     - K : α [T - T0]
        #     - τ K : α [∂T/∂t + (a·∇)T]
        #     - τ K : ∂α/∂T [∂T/∂t + (a·∇)T] [T - T0]
        #
        # where `a` is the velocity of the co-moving frame, as measured in the
        # laboratory frame.
        #
        # Finally, collecting terms, our Eulerian constitutive law is
        #
        #   σ = K : ε + τ K dε/dt
        #     - K : α [T - T0]
        #     - τ [K : α + K : ∂α/∂T [T - T0]] [∂T/∂t + (a·∇)T]
        #
        # Note that the contribution from temperature-varying `α` is nonlinear
        # in `T`, if `T` varies in time; or if `T` varies in space and there is
        # axial motion (`a ≠ 0`).
        #
        # Note also that there is no need to differentiate the stiffness tensor `K`;
        # we can just as well take `K = K(T)` without affecting the equations
        # (although, as a practical consideration, this does affect the API).
        #
        # Since for an isotropic solid, the elastic stiffness tensor `K` is
        #
        #   K = 2 μ ES + 3 λ EV
        #
        # we have
        #
        #   K : α = 2 μ symm(α) + 3 λ vol(α)
        #         = 2 μ symm(α) + λ I tr(α)
        #
        # and similarly for `K : ∂α/∂T`.
        #
        # Now, for simplicity, consider first the elastic case (τ = 0).
        # In the primal formulation, the stress is written out as (in 2D and 3D)
        #
        #   σik = 2 μ [(1/2) (∂i uk + ∂k ui)] + λ δik ∂m um
        #       - [2 μ [(1/2) (αik + αki)] + λ δik αmm] [T - T0]
        #
        # The normal component of the stress is
        #
        #   ni σik = 2 μ [(1/2) (ni ∂i uk + ni ∂k ui)] + λ ni δik ∂m um
        #          - [2 μ [(1/2) (ni αik + ni αki)] + λ ni δik αmm] [T - T0]
        #          = μ [ni ∂i uk + ni ∂k ui] + λ nk ∂m um
        #          - [μ [ni αik + ni αki] + λ nk αmm] [T - T0]
        #
        # which in nabla notation reads
        #
        #   n·σ = μ (n·∇)u + μ [∇u]·n + λ n tr(ε(u))
        #       - [μ n·α + μ α·n + λ n tr(α)] [T - T0]
        #
        # or alternatively,
        #
        #   n·σ = μ (n·∇)u + μ n·transpose(∇u) + λ n tr(ε(u))
        #       - [μ n·α + μ n·transpose(α) + λ n tr(α)] [T - T0]
        #
        # Dot-multiplying this by the test `w`, to set a BC on `(n·∇)u`,
        # in the weak form we must include the terms
        #
        #   - [μ du0dn + μ n·transpose(∇u) + λ tr(ε(u)) n]·w ds
        #   + [μ n·α + μ n·transpose(α) + λ tr(α) n]·w [T - T0] ds
        #
        # where `du0dn` is the prescribed value for `(n·∇)u`. The signs
        # come from the original term `-∇·σ`, so that we need `-n·σ`.
        #
        # Similarly, in the Kelvin-Voigt version, we have
        #
        #   n·σ = μ (n·∇)u + μ [∇u]·n + λ n tr(ε(u))
        #       + τ [μ (n·∇)v + μ [∇v]·n + λ n tr(ε(v))]
        #       - [μ n·α + μ α·n + λ n tr(α)] [T - T0]
        #       - τ [μ n·α + μ α·n + λ n tr(α)] dT/dt
        #       - τ [μ n·(dα/dt) + μ (dα/dt)·n + λ n tr(dα/dt)] [T - T0]
        #
        # so in the weak form we must include the terms
        #
        #   - [μ du0dn + μ n·transpose(∇u) + λ tr(ε(u)) n]·w ds
        #   - τ [μ dv0dn + μ n·transpose(∇v) + λ tr(ε(v)) n]·w ds
        #   + [μ n·α + μ n·transpose(α) + λ tr(α) n]·w [T - T0] ds
        #   + τ [μ n·α + μ n·transpose(α) + λ tr(α) n]·w dT/dt ds
        #   + τ [μ n·dα/dt + μ n·transpose(dα/dt) + λ tr(dα/dt) n]·w [T - T0] ds
        #
        # or, taking `α = α(T)`, expanding the `dα/dt`, and collecting,
        #
        #   - [μ du0dn + μ n·transpose(∇u) + λ tr(ε(u)) n]·w ds
        #   - τ [μ dv0dn + μ n·transpose(∇v) + λ tr(ε(v)) n]·w ds
        #   + [μ n·α + μ n·transpose(α) + λ tr(α) n]·w [T - T0] ds
        #   + τ [ μ n·α + μ n·transpose(α) + λ tr(α) n
        #       + [μ n·∂α/∂T + μ n·transpose(∂α/∂T) + λ tr(∂α/∂T) n] [T - T0]
        #       ]·w [∂T/∂t + (a·∇)T] ds
        #
        # where `du0dn` is the prescribed value for `(n·∇)u`,
        # and `dv0dn` is the prescribed value for `(n·∇)v`.
        #
        # TODO: Make the solver work also in 1D, to compare to previous results.
        # TODO: We will have different boundary conditions. May be easiest to
        # TODO: change the BCs in the 1D analysis.
        #
        # In 1D, the stress for thermoviscoelastic Kelvin-Voigt is
        #
        #   σ = E [ ∂u/∂x + τ ∂v/∂x
        #         + α [T - T0]
        #         + τ [α + ∂α/∂T [T - T0]] [∂T/∂t + V0 ∂T/∂x]]
        #
        # so at the boundaries (domain endpoints)
        #
        #   n σ = n E [ ∂u/∂x + τ ∂v/∂x
        #             + α [T - T0]
        #             + τ [α + ∂α/∂T [T - T0]] [∂T/∂t + V0 ∂T/∂x]]
        #
        # where `n = ±1`. So in the weak form, we must add the terms
        #
        #   -n E [ du0dn + τ dv0dn
        #        + α [T - T0]
        #        + τ [α + ∂α/∂T [T - T0]] [∂T/∂t + V0 ∂T/∂x]] w ds
        #
        # Note that there is now just one term for linear elasticity, one term for
        # the viscous part of Kelvin-Voigt, one term for thermoelastic effects,
        # and two terms (one of which nonlinear in `T`) for thermoviscous effects.
        #
        # When solving the steady state on pen and paper, `v = V0 ∂u/∂x`, so in terms
        # of `u`, this actually sets boundary conditions on `∂u/∂x` and `∂²u/∂x²`.
        #
        # In the 1D steady state *without* thermal effects, the condition `n σ = 0`
        # is equivalent with `∂u/∂x = ∂²u/∂x² = 0`; in general, these will differ
        # (due to the thermal contribution to the strain).
        #
        # The main practical complication here is the API. We must be able to specify
        # for only *some* boundaries with Neumann BCs (namely, the outlet) to use the
        # normal gradient BC, whereas others (the free edges) should use the stress
        # BC. So we must split `ds` to apply BCs selectively by boundary tag, and
        # include a list of boundary tags in the Neumann BC specification.
        #
        # See the tutorial:
        #   https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html#fenics-implementation-14
        # particularly, how to redefine the measure `ds` in terms of boundary markers:
        #
        #   ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundary_parts)
        #
        # For this, we need to pass `boundary_parts` from the main script into the
        # solver. Maybe also start using a dictionary (like the tutorial does) for
        # defining the BCs; this is more flexible when there are several options.

        # Constitutive equation
        Id = Identity(ε(u).geometric_dimension())
        K_inner = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`
        if self.τ == 0.0:  # Linear elastic (LE)
            # for equation
            Σ = K_inner(ε(u))
            # for visualization
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u_)), sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            Σ = K_inner(ε(u)) + τ * K_inner(ε(v))
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner(ε(u_)) + τ * K_inner(ε(v_)), sym(φ)) * dx)

        # Primal formulation allows using a Neumann BC for stress.
        Σ0 = self.bcσ
        F_u = (ρ * advw(a, v, w, n) +
               inner(Σ.T, ε(w)) * dx - dot(dot(n, Σ0.T), w) * ds -
               ρ * dot(b, w) * dx)
        F_v = (dot(v, ψ) * dx - advw(a, u, ψ, n))

        # SUPG: streamline upwinding Petrov-Galerkin. The residual is evaluated elementwise in strong form.
        #
        # Seems that for this equation system, the skew-symmetric advection already stabilizes enough;
        # for some reason, adding in SUPG makes these equations more unstable. Tuning doesn't help.
        #
        # deg = Constant(self.V.ufl_element().degree())
        # moo = Constant(max(self.λ, 2 * self.μ, self.τ * self.λ, self.τ * 2 * self.μ))
        # τ_SUPG = (α0 / deg) * (2 * mag(a) / he + 4 * (moo / ρ) / he**2)**-1  # [τ] = s
        # R = (ρ * advs(a, v) - div(Σ) - ρ * b)
        # F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, w), R) * dx
        # F_u += F_SUPG
        #
        # # Especially this seems to destabilize significantly.
        # deg = Constant(self.V.ufl_element().degree())
        # τ_SUPG = (α0 / deg) * (2 * mag(a) / he)**-1  # [τ] = s
        # R = (v - advs(a, u))
        # F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        # F_v += F_SUPG

        F = F_u + F_v
        self.a = lhs(F)
        self.L = rhs(F)

        # For visualization only.
        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)

        εu = self.εu  # unknown
        εv = self.εv
        q = self.q
        F_εu = inner(εu, q) * dx - inner(ε(self.u_), sym(q)) * dx
        F_εv = inner(εv, q) * dx - inner(ε(self.v_), sym(q)) * dx
        self.a_εu = lhs(F_εu)
        self.L_εu = rhs(F_εu)
        self.a_εv = lhs(F_εv)
        self.L_εv = rhs(F_εv)

    def solve(self) -> typing.Tuple[int, int, typing.Tuple[int, float]]:
        """Solve the steady state.

        The solution becomes available in `self.s_`.
        """
        begin("Solve steady state")
        A = assemble(self.a)
        b = assemble(self.L)
        [bc.apply(A) for bc in self.bcu]
        [bc.apply(b) for bc in self.bcu]
        [bc.apply(A) for bc in self.bcv]
        [bc.apply(b) for bc in self.bcv]
        if not self.bcu:
            A_PETSc = as_backend_type(A)
            A_PETSc.set_near_nullspace(self.null_space)
            A_PETSc.set_nullspace(self.null_space)
            self.null_space.orthogonalize(b)
        it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')

        # VISUALIZATION PURPOSES ONLY
        A2a = assemble(self.a_εu)
        b2a = assemble(self.L_εu)
        solve(A2a, self.εu_.vector(), b2a, 'bicgstab', 'sor')
        A2b = assemble(self.a_εv)
        b2b = assemble(self.L_εv)
        solve(A2b, self.εv_.vector(), b2b, 'bicgstab', 'sor')
        A2c = assemble(self.a_σ)
        b2c = assemble(self.L_σ)
        solve(A2c, self.σ_.vector(), b2c, 'bicgstab', 'sor')

        end()
        return it

# --------------------------------------------------------------------------------

def step_adaptive(solver: typing.Union[EulerianSolid, EulerianSolidAlternative],
                  max_substeps: int = 16):
    """Simple adaptive timestepping for `EulerianSolid` and `EulerianSolidAlternative`.

    If the solver does not converge at the original `solver.dt`, halve the timestep size
    (and double the number of substeps) until it converges, or until `max_substeps` is
    exceeded.

    The results are always reported at the end of the original `solver.dt`, so to the outside,
    the result looks as if the solver succeeded at the original timestep size.

    This is mainly useful as a workaround for temporary convergence issues during a simulation,
    or when the first few timesteps at the very beginning of a simulation need to be smaller
    (so that the wall time gain from the larger timestep outweighs the amortized cost of the
    slow start).

    In case the solver always fails to converge at the original dt, this is much slower than
    just using a smaller dt to begin with. This will always try solving with the original dt
    first; running the maximum number of system iterations to detect a convergence failure
    is very slow.
    """
    # import dolfin  # DEBUG

    @contextmanager
    def timestep_temporarily_changed_to(dt):
        old_dt = solver.dt
        old_un = Function(solver.V)
        old_vn = Function(solver.V)
        old_σn = Function(solver.Q)
        old_un.assign(solver.u_n)
        old_vn.assign(solver.v_n)
        old_σn.assign(solver.σ_n)
        try:
            solver.dt = dt
            yield
        finally:
            solver.dt = old_dt
            solver.u_n.assign(old_un)
            solver.v_n.assign(old_vn)
            solver.σ_n.assign(old_σn)

    def solve_in_n_substeps(n):
        substep_sysits = []
        substep_it1s = []
        substep_it2s = []
        substep_it3s = []
        for k in range(n):
            it1, it2, it3, (sysit, e) = solver.step()
            substep_it1s.append(it1)
            substep_it2s.append(it2)
            substep_it3s.append(it3)
            substep_sysits.append(sysit)
            # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
            #     print(f"substep {k}, sysit = {sysit}, e = {e:0.6g}")
            if e >= solver.tol:  # timestep still too large; cancel
                return sum(substep_it1s), sum(substep_it2s), sum(substep_it3s), (sum(substep_sysits), e)
            solver.commit()  # temporarily; the context manager rolls this back
        return sum(substep_it1s), sum(substep_it2s), sum(substep_it3s), (sum(substep_sysits), e)

    class Converged(Exception):
        pass

    # Try first with the original dt
    n = 1
    dt = solver.dt
    sysits = []
    it1s = []
    it2s = []
    it3s = []
    it1, it2, it3, (sysit, e) = solver.step()
    it1s.append(it1)
    it2s.append(it2)
    it3s.append(it3)
    sysits.append(sysit)

    if e > solver.tol:
        try:
            n = 2
            dt = solver.dt / n
            min_dt = solver.dt / 16
            while e > solver.tol:
                # if dolfin.MPI.comm_world.rank == 0:  # DEBUG
                #     print(f"No convergence at {int(n / 2)} substeps at dt={2 * dt:0.6g} s (e = {e:0.6g} > tol = {self.tol:0.6g}); trying again with {n} substeps at dt={dt:0.6g} s")

                # IMPORTANT: After failed attempt, restore the initial guess to the original old field
                solver.u_.assign(solver.u_n)
                solver.v_.assign(solver.v_n)
                solver.σ_.assign(solver.σ_n)

                with timestep_temporarily_changed_to(dt):
                    it1, it2, it3, (sysit, e) = solve_in_n_substeps(n)
                    it1s.append(it1)
                    it2s.append(it2)
                    it3s.append(it3)
                    sysits.append(sysit)
                    if e < solver.tol:
                        raise Converged
                    if dt <= min_dt:
                        raise RuntimeError(f"No convergence at smallest allowed adaptive timestep {min_dt:0.6g} s (e = {e:0.6g} > tol = {solver.tol:0.6g})")
                    n *= 2
                    dt /= 2
        except Converged:
            pass

    return sum(it1s), sum(it2s), sum(it3s), (sum(sysits), e), (n, dt)
