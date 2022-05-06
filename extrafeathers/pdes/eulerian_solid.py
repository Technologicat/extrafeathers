# -*- coding: utf-8; -*-
"""
Axially moving solid, Eulerian view, small-displacement regime (on top of axial motion).

  ρ dV/dt - ∇·transpose(σ) = ρ b

where `V` is the material parcel velocity, and `d/dt` is the material derivative.
This is approximated as

  dV/dt ≈ [∂²u/∂t² + 2 (a·∇) ∂u/∂t + (a·∇)(a·∇)u]

where `a` is the (constant) velocity of the co-moving frame.

`u` is then the displacement in the laboratory frame, on top of the imposed
constant-velocity axial motion.

We solve in mixed form, because for Kelvin-Voigt and SLS, the axial motion
introduces a third derivative in the strong form (in the mixed form,
appearing as a spatial derivative of ε in the constitutive equation for σ).
"""

__all__ = ["EulerianSolid", "SteadyStateEulerianSolid"]

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
from .util import ufl_constant_property, StabilizerFlags


def ε(u):
    """Symmetric gradient of the displacement `u`, a.k.a. the infinitesimal (Cauchy) strain.

        (symm ∇)(u) = (1/2) (∇u + transpose(∇u))
    """
    return sym(nabla_grad(u))

# def vol(T):
#     """Volumetric part of rank-2 tensor `T`."""
#     return 1 / 3 * Identity(T.geometric_dimension()) * tr(T)


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

    References:
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

        # Set up the null space. We'll remove it in the Krylov solver.
        #
        # https://fenicsproject.discourse.group/t/rotation-in-null-space-for-elasticity/4083
        # https://bitbucket.org/fenics-project/dolfin/src/946dbd3e268dc20c64778eb5b734941ca5c343e5/python/demo/undocumented/elasticity/demo_elasticity.py#lines-35:52
        # https://bitbucket.org/fenics-project/dolfin/issues/587/functionassigner-does-not-always-call
        #
        # Null space of the linear momentum balance is {u: ε(u) = 0 and ∇·u = 0}
        # This consists of rigid-body translations and infinitesimal rigid-body rotations.

        # Strictly, this is the null space of linear elasticity, but the physics shouldn't
        # be that much different for the other linear models.
        dim = self.mesh.topology().dim()
        if dim == 2:
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

        def advw(a, p, q):
            """Advection operator, weak form.

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected
            `q`: test function of the quantity `p`

            `p` and `q` must be at least C0.
            """
            return ((1 / 2) * (dot(dot(a, nabla_grad(p)), q) -
                               dot(dot(a, nabla_grad(q)), p)) * dx +
                               (1 / 2) * dot(n, a) * dot(p, q) * ds)
        def advs(a, p):
            """Advection operator, strong form (for SUPG residual).

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected

            `a` and `p` must be at least C0.
            """
            return dot(a, nabla_grad(p)) + (1 / 2) * div(a) * p

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
        # TODO:
        #  - Add elastothermal effects:  ∫ φ : [KE : α] [T - T0] dΩ  (same sign as ∫ φ : KE : ε dΩ term)
        #    - Need a FEM field for temperature T, and parameters α and T0
        #  - Add viscothermal effects, see eq. (768) in report
        #  - Orthotropic linear elastic
        #  - Orthotropic Kelvin-Voigt
        #  - Isotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #  - Orthotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)

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
        K_inner_operator = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`
        K_inner_εu_ = K_inner_operator(εu_)
        K_inner_εv_ = K_inner_operator(εv_)

        # # Original definitions for step 2, no projection
        # εu_ = ε(U)
        # εv_ = ε(V)
        # Id = Identity(εu_.geometric_dimension())
        # K_inner_operator = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`
        # K_inner_εu_ = K_inner_operator(εu_)
        # K_inner_εv_ = K_inner_operator(εv_)

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
                   inner(K_inner_εu_, sym(φ)) * dx)
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
            # def mag(vec):
            #     return dot(vec, vec)**(1 / 2)
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
            #        (inner(K_inner_εu_ + τ * K_inner_εv_, sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
            #         τ * dot(a, n) * inner(K_inner_εu_, sym(φ)) * ds -  # generated by ∫ (φ:Kη):(a·∇)ε dx
            #         τ * inner(K_inner_εu_, advs(a, sym(φ))) * dx))  # generated by ∫ (φ:Kη):(a·∇)ε dx

            # # Look, ma, no integration by parts.
            # F_σ = (inner(Σ, φ) * dx -
            #        (inner(K_inner_εu_ + τ * K_inner_εv_, sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
            #         τ * inner(K_inner_operator(advs(a, εu_)), sym(φ)) * dx))  # ∫ (φ:Kη):(a·∇)ε dx

            # "Symmetrized": half integrated by parts, half not.
            # The `a·∇` operates on both the quantity and test parts. This is most similar to the
            # skew-symmetric advection operator used in modern FEM discretizations of Navier-Stokes.
            F_σ = (inner(Σ, φ) * dx -
                   (inner(K_inner_εu_ + τ * K_inner_εv_, sym(φ)) * dx +  # linear elastic and τ ∂/∂t (...)
                    0.5 * τ * inner(K_inner_operator(advs(a, εu_)), sym(φ)) * dx -  # 1/2 ∫ (φ:Kη):(a·∇)ε dx
                    0.5 * τ * inner(K_inner_εu_, advs(a, sym(φ))) * dx +  # 1/2 ∫ (φ:Kη):(a·∇)ε dx
                    0.5 * τ * dot(a, n) * inner(K_inner_εu_, sym(φ)) * ds))  # generated, 1/2 ∫ (φ:Kη):(a·∇)ε dx

            # TODO: Fix the stress stabilization. Commented out, because doesn't seem to work correctly.
            #
            # # SUPG stabilization. Note that the SUPG terms are added only in the element interiors.
            # τ_SUPG = (α0 / self.Q.ufl_element().degree()) * (1 / (θ * dt) + 2 * mag(a) / he)**-1  # [τ] = s  # TODO: tune value
            # # τ_SUPG = (α0 / self.Q.ufl_element().degree()) * (2 * mag(a) / he)**-1  # [τ] = s  # TODO: tune value
            # # τ_SUPG = Constant(0.004)  # TODO: tune value
            # # The residual is evaluated elementwise in strong form, at the end of the timestep.
            # εu_ = ε(u_)
            # εv_ = ε(v_)
            # K_inner_εu_ = K_inner_operator(εu_)
            # K_inner_εv_ = K_inner_operator(εv_)
            # # # We need advs(a, εu_), but if `u` uses a degree-1 basis, ∇εu_ = 0. Thus, we need to
            # # # project εu_ into Q before differentiating to get a useful derivative in the element interiors.
            # # # Use a temporary storage (εtmp) for that; the iteration loop fills in the actual data.
            # # R = σ - (K_inner_εu_ + τ * (K_inner_εv_ + K_inner_operator(advs(a, self.εtmp))))
            # # OTOH, in Navier-Stokes, some of the terms may be zero and SUPG works just fine.
            # R = σ - (K_inner_εu_ + τ * (K_inner_εv_ + K_inner_operator(advs(a, εu_))))
            # # Same here; the RHS is symmetric, so symmetrize the test function.
            # F_SUPG = enable_SUPG_flag * τ_SUPG * inner(advs(a, sym(φ)), R) * dx
            # F_σ += F_SUPG

            # Let's try classic artificial diffusion along streamlines? (This didn't help, either.)
            # F_σ += he**2 / mag(a)**2 * inner(dot(a, nabla_grad(σ)), dot(a, nabla_grad(sym(φ)))) * dx

        # Step 3: solve `v` from momentum equation
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
               2 * ρ * advw(a, V, ψ) -
               ρ * dot(dot(a, nabla_grad(U)), dot(a, nabla_grad(ψ))) * dx +  # from +∫ ρ [(a·∇)(a·∇)u]·ψ dx
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(U)), ψ)) * ds +
               inner(Σ.T, ε(ψ)) * dx -
               dot(dot(n, Σ), ψ) * ds -
               ρ * dot(b, ψ) * dx)

        # SUPG: streamline upwinding Petrov-Galerkin.
        def mag(vec):
            return dot(vec, vec)**(1 / 2)
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

class SteadyStateEulerianSolid:
    """Axially moving linear solid, small-displacement Eulerian formulation.

    Like `EulerianSolid`, but steady state.

    Diriclet BCs are now given for `u` (NOTE!) and `σ`.

    Note `v = ∂u/∂t ≡ 0`, because we are in an Eulerian steady state.
    """
    def __init__(self, V: VectorFunctionSpace, Q: TensorFunctionSpace,
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

        self.u, self.σ = u, σ  # trials
        self.ψ, self.φ = ψ, φ  # tests
        self.u_, self.σ_ = u_, σ_  # solution

        self.S = S
        self.s_ = s_

        # Strictly, this is the null space of linear elasticity, but the physics shouldn't
        # be that much different for the other linear models.
        dim = self.mesh.topology().dim()
        if dim == 2:
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

        def advw(a, p, q):
            """Advection operator, weak form.

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected
            `q`: test function of the quantity `p`

            `p` and `q` must be at least C0.
            """
            return ((1 / 2) * (dot(dot(a, nabla_grad(p)), q) -
                               dot(dot(a, nabla_grad(q)), p)) * dx +
                               (1 / 2) * dot(n, a) * dot(p, q) * ds)
        def advs(a, p):
            """Advection operator, strong form (for SUPG residual).

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected

            `a` and `p` must be at least C0.
            """
            return dot(a, nabla_grad(p)) + (1 / 2) * div(a) * p

        # Define variational problem
        #
        # We build one monolithic equation.

        # Constitutive equation
        #
        # TODO:
        #  - Add elastothermal effects:  ∫ φ : [KE : α] [T - T0] dΩ  (same sign as ∫ φ : KE : ε dΩ term)
        #    - Need a FEM field for temperature T, and parameters α and T0
        #  - Add viscothermal effects, see eq. (768) in report
        #  - Orthotropic linear elastic
        #  - Orthotropic Kelvin-Voigt
        #  - Isotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #  - Orthotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)

        εu = ε(u)  # NOTE: Based on the *unknown* `u`.
        Id = Identity(εu.geometric_dimension())
        K_inner_operator = lambda ε: 2 * μ * ε + λ * Id * tr(ε)  # `K:(...)`
        K_inner_εu_ = K_inner_operator(εu)

        # Choose constitutive model
        if self.τ == 0.0:  # Linear elastic (LE)
            F_σ = (inner(σ, φ) * dx -
                   inner(K_inner_εu_, sym(φ)) * dx)
        else:  # Axially moving Kelvin-Voigt (KV)
            # TODO: This equation may need SUPG when τ ≠ 0, a ≠ 0.
            F_σ = (inner(σ, φ) * dx -
                   (inner(K_inner_εu_, sym(φ)) * dx +  # linear elastic
                    τ * dot(a, n) * inner(K_inner_εu_, sym(φ)) * ds -  # generated by ∫ (φ:Kη):(a·∇)ε dx
                    τ * inner(K_inner_εu_, advs(a, sym(φ))) * dx))  # generated by ∫ (φ:Kη):(a·∇)ε dx

        # Linear momentum balance
        #
        # This equation has no `u` when `a = 0`, so we use a monolithic formulation
        # (both equations solved as one system); this allows the solver to see the
        # dependence `σ = σ(u)`.
        F_u = (-ρ * dot(dot(a, nabla_grad(u)), dot(a, nabla_grad(ψ))) * dx +  # from +∫ ρ [(a·∇)(a·∇)u]·ψ dx
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(u)), ψ)) * ds +
               inner(σ.T, ε(ψ)) * dx -
               dot(dot(n, σ), ψ) * ds -
               ρ * dot(b, ψ) * dx)

        F = F_σ + F_u
        self.a = lhs(F)
        self.L = rhs(F)

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
        return it
