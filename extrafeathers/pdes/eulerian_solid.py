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

__all__ = ["EulerianSolid"]

from itertools import chain
import typing

from fenics import (VectorFunctionSpace, TensorFunctionSpace, MixedElement, FunctionSpace,
                    split, FunctionAssigner,
                    TrialFunctions, TestFunctions,
                    Constant, Expression, Function,
                    FacetNormal, DirichletBC,
                    dot, inner, outer, sym, tr,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve, normalize,
                    interpolate, VectorSpaceBasis, as_backend_type,
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


# TODO: set initial condition?
# TODO: use nondimensional form
class EulerianSolid:
    """TODO: document this

    Time integration is performed using the θ method; Crank-Nicolson by default.

    `V`: function space for displacement
    `Q`: function space for stress

         Note `σ` is based on `ε`, which is essentially the gradient of `u`.
         Therefore, for consistency, the polynomial order of the space `Q`
         should be one lower than that of `V`.

    `ρ`: density [kg / m³]
    `λ`: Lamé's first parameter [Pa]
    `μ`: shear modulus [Pa]
    `V0`: velocity of co-moving frame in +x direction (constant) [m/s]
    `bcu`: Dirichlet boundary conditions for displacement
    `bcσ`: Dirichlet boundary conditions for stress
    `dt`: timestep [s]
    `θ`: theta-parameter for the time integrator, θ ∈ [0, 1].
         Default 0.5 is Crank-Nicolson; 0 is forward Euler, 1 is backward Euler.

         Note that for θ = 0, the SUPG stabilization parameter τ_SUPG → 0,
         so when using forward Euler, it does not make sense to enable the
         SUPG stabilizer.

    As the mesh, we use `V.mesh()`; both `V` and `Q` must be defined on the same mesh.
    """
    def __init__(self, V: VectorFunctionSpace, Q: TensorFunctionSpace,
                 ρ: float, λ: float, μ: float,
                 V0: float,
                 bcu: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Trial and test functions
        #
        # `u`: displacement
        # `v`: Eulerian time rate of displacement (reduction to 1st order system)
        #  - `v` is based on ∂u/∂t, so the polynomial order of the space
        #    used for approximating `v` should be the same as that for `u`.
        #    We can just use another copy of the space `V`.
        # `σ`: stress
        #  - Mixed formulation; stress has its own equation to allow easily
        #   changing the constitutive model.
        #  - Also, need to treat it this way for an Eulerian description of
        #    viscoelastic models (the material derivative introduces a term
        #    that is one order of ∇ higher).
        #  - `σ` is based on `ε`, which is essentially the gradient of `u`.
        #    Therefore, for consistency, the polynomial order of the space `Q`
        #    should be one lower than that of `V`.
        e = MixedElement(V.ufl_element(), V.ufl_element(), Q.ufl_element())
        S = FunctionSpace(self.mesh, e)
        u, v, σ = TrialFunctions(S)  # no suffix: UFL symbol for unknown quantity
        ψ, w, φ = TestFunctions(S)
        s_ = Function(S)  # suffix _: latest computed approximation
        s_n = Function(S)  # suffix _n: old value (end of previous timestep)
        u_, v_, σ_ = split(s_)
        u_n, v_n, σ_n = split(s_n)

        # Set up the null space. We'll remove it in the Krylov solver.
        #
        # https://fenicsproject.discourse.group/t/rotation-in-null-space-for-elasticity/4083
        # https://bitbucket.org/fenics-project/dolfin/src/946dbd3e268dc20c64778eb5b734941ca5c343e5/python/demo/undocumented/elasticity/demo_elasticity.py#lines-35:52
        # https://bitbucket.org/fenics-project/dolfin/issues/587/functionassigner-does-not-always-call
        #
        # `u`: null space of the linear momentum balance is {u: ε(u) = 0 and ∇·u = 0}
        # This consists of rigid-body translations and infinitesimal rigid-body rotations.
        #
        # Strictly, this is the null space of linear elasticity, but the physics shouldn't
        # be that much different for the other linear models.
        dim = self.mesh.topology().dim()
        if dim == 2:
            fus = [Constant((1, 0)),
                   Constant((0, 1)),
                   Expression(("x[1]", "-x[0]"), degree=1)]
        elif dim == 3:
            fus = [Constant((1, 0, 0)),
                   Constant((0, 1, 0)),
                   Constant((0, 0, 1)),
                   Expression(("0", "x[2]", "-x[1]"), degree=1),  # around x axis
                   Expression(("-x[2]", "0", "x[0]"), degree=1),  # around y axis
                   Expression(("x[1]", "-x[0]", "0"), degree=1)]  # around z axis
        else:
            raise NotImplementedError(f"dim = {dim}")
        # `v`, `σ`:
        #   - No null space; in each equation, there is a reaction term.

        # If our function space was just `V`, we could just:
        #     null_space_basis = [interpolate(fu, V).vector() for fu in fus]
        # But we have a mixed formulation, so we must insert zero functions
        # for the other fields:
        zeroV = Function(V)
        zeroV.vector()[:] = 0.0
        zeroQ = Function(Q)
        zeroQ.vector()[:] = 0.0
        assigner = FunctionAssigner([V, V, Q], S)
        fss = [Function(S) for _ in range(len(fus))]
        for fu, fs in zip(fus, fss):
            assigner.assign([interpolate(fu, V), zeroV, zeroQ], fs)
        null_space_basis = [fs.vector() for fs in fss]
        [normalize(vec, 'l2') for vec in null_space_basis]  # TODO: normalize full vector or sub only?

        # # May be needed in some FEniCS versions to avoid PETSc error in `VecCopy`
        # for vec in null_space_basis:
        #     vec.apply("insert")

        basis = VectorSpaceBasis(null_space_basis)
        # basis.orthonormalize()  # TODO: for some reason this says there is a linear dependence in the basis?
        self.null_space = basis

        self.S = S
        self.s_, self.s_n = s_, s_n

        self.V = V
        self.Q = Q

        self.u, self.v, self.σ = u, v, σ  # trials
        self.ψ, self.w, self.φ = ψ, w, φ  # tests
        self.u_, self.v_, self.σ_ = u_, v_, σ_  # last computed approximation
        self.u_n, self.v_n, self.σ_n = u_n, v_n, σ_n  # previous value

        # Dirichlet boundary conditions
        self.bcu = bcu
        self.bcσ = bcσ

        # Local mesh size (for stabilization terms)
        self.he = cell_mf_to_expression(meshsize(self.mesh))

        # Velocity of co-moving frame (constant; to generalize,
        # need to update formulation to include fictitious forces)
        self.a = Constant((V0, 0))

        # Specific body force. FEM function for maximum generality.
        self.b = Function(V)
        self.b.vector()[:] = 0.0  # placeholder value

        # Parameters.
        # TODO: use FEM fields, we will need these to be temperature-dependent.
        # TODO: parameterize using the (rank-4) stiffness/viscosity tensors
        #       (better for arbitrary symmetry group)
        self._ρ = Constant(ρ)
        self._λ = Constant(λ)
        self._μ = Constant(μ)
        self._dt = Constant(dt)
        self._θ = Constant(θ)

        # Numerical stabilizer on/off flags.
        self.stabilizers = EulerianSolidStabilizerFlags()

        # SUPG stabilizer tuning parameter.
        self._α0 = Constant(1)

        self.compile_forms()

    ρ = ufl_constant_property("ρ", doc="Density [kg / m³]")
    λ = ufl_constant_property("λ", doc="Lamé's first parameter [Pa]")
    μ = ufl_constant_property("μ", doc="Shear modulus [Pa]")
    dt = ufl_constant_property("dt", doc="Timestep [s]")
    θ = ufl_constant_property("θ", doc="Time integration parameter of θ method")
    α0 = ufl_constant_property("α0", doc="SUPG stabilizer tuning parameter")

    def compile_forms(self) -> None:
        n = FacetNormal(self.mesh)

        # Displacement
        u = self.u      # new (unknown)
        ψ = self.ψ      # test
        # u_ = self.u_    # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Eulerian time rate of displacement,  v = ∂u/∂t
        v = self.v
        w = self.w
        # v_ = self.v_
        v_n = self.v_n

        # Stress
        σ = self.σ
        φ = self.φ
        # σ_ = self.σ_
        σ_n = self.σ_n

        # Specific body force
        b = self.b

        # Local mesh size (for stabilization terms)
        he = self.he

        # Parameters
        ρ = self._ρ
        λ = self._λ
        μ = self._μ
        dt = self._dt
        θ = self._θ
        α0 = self._α0

        enable_SUPG_flag = self.stabilizers._SUPG

        # We will time-integrate using the θ method.
        U = (1 - θ) * u_n + θ * u
        V = (1 - θ) * v_n + θ * v
        Σ = (1 - θ) * σ_n + θ * σ
        dudt = (u - u_n) / dt
        dvdt = (v - v_n) / dt
        # dσdt = (σ - σ_n) / dt  # Kelvin-Voigt, SLS

        a = self.a  # convection velocity

        def advw(a, u, ψ):
            """Advection operator, weak form.

            `a`: advection velocity (assumed divergence-free)
            `u`: quantity being advected
            `ψ`: test function of `u`
            """
            return ((1 / 2) * (dot(dot(a, nabla_grad(u)), ψ) -
                               dot(dot(a, nabla_grad(ψ)), u)) * dx +
                               (1 / 2) * dot(n, a) * dot(u, ψ) * ds)
        def advs(a, u):
            """Advection operator, strong form (for SUPG residual).

            `a`: advection velocity (assumed divergence-free)
            `u`: quantity being advected
            """
            return dot(a, nabla_grad(u)) + (1 / 2) * div(a) * u

        # Define variational problem
        #
        # - Valid boundary conditions:
        #   - Displacement boundary: `u` given, no condition on `σ`
        #     - Dirichlet boundary for `u`; those rows of `F_u` removed, ∫ ds terms don't matter.
        #     - Affects `σ` automatically, via the stress expression.
        #   - Stress (traction) boundary: `σ` given, no condition on `u`
        #     - Dirichlet boundary for `σ`; those rows of `F_σ` removed.
        #     - Need to include the -∫ n·[σ·ψ] ds term in `F_u`.
        #   - No boundary conditions on `v` in any case (auxiliary variable, no spatial derivatives).
        # - `F_v` and `F_σ` have no boundary integrals.
        F_u = (ρ * dot(dvdt, ψ) * dx +
               2 * ρ * advw(a, V, ψ) -
               ρ * dot(dot(a, nabla_grad(U)), dot(a, nabla_grad(ψ))) * dx +
               inner(Σ, ε(ψ)) * dx -
               dot(n, dot(Σ, ψ)) * ds +
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(U)), ψ)) * ds -  # +∫ ρ ([a⊗a]·∇u)·ψ dx
               ρ * dot(b, ψ) * dx)
        F_v = dot(V - dudt, w) * dx  # v = ∂u/∂t

        # TODO:
        #  - Add elastothermal effects:  ∫ φ : [KE : α] [T - T0] dΩ  (same sign as ∫ φ : KE : ε dΩ term)
        #    - Need a FEM field for temperature T, and parameters α and T0
        #  - Orthotropic linear elastic
        #  - Isotropic Kelvin-Voigt, includes dε/dt = ∂ε/∂t + (a·∇)ε
        #    - Add viscothermal effects, see eq. (768) in report
        #  - Orthotropic Kelvin-Voigt, includes dε/dt = ∂ε/∂t + (a·∇)ε
        #  - Isotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #  - Orthotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #
        #   σ = K : ε
        #   K = 2 μ ES + 3 λ EV
        # Because for any rank-2 tensor T,
        #   ES : T = symm(T)
        #   EV : T = vol(T) = (1/3) I tr(T)
        # we have
        #   σ = 2 μ symm(ε) + 3 λ vol(ε)
        #     = 2 μ ε + 3 λ vol(ε)
        # where on the last line we have used the symmetry of ε.
        # σ = 2 μ ε(u) + 3 λ vol(u) = 2 μ ε(u) + λ I tr(u)
        εu = ε(U)
        stress_expr = 2 * μ * εu + λ * Identity(εu.geometric_dimension()) * tr(εu)
        F_σ = inner(Σ - stress_expr, φ) * dx

        F = F_u + F_v + F_σ

        # SUPG: streamline upwinding Petrov-Galerkin.
        #
        def mag(vec):
            return dot(vec, vec)**(1 / 2)
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (μ / ρ) / he**2)**-1  # [τ] = s  # TODO: tune value
        # The residual is evaluated elementwise in strong form,
        # at the end of the timestep.
        R = (ρ * ((v - v_n) / dt + 2 * advs(a, v) + advs(a, advs(a, u))) -
             div(σ) - ρ * b)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        F += F_SUPG

        self.a = lhs(F)
        self.L = rhs(F)

    def step(self) -> typing.Tuple[int, int, int]:
        """Take a timestep of length `self.dt`.

        Updates the latest computed solution.
        """
        begin("Solve timestep")
        A = assemble(self.a)
        b = assemble(self.L)
        # When Neumann BCs only, eliminate rigid-body motions
        if not self.bcu:
            as_backend_type(A).set_nullspace(self.null_space)
            self.null_space.orthogonalize(b)
        [bc.apply(A) for bc in chain(self.bcu, self.bcσ)]
        [bc.apply(b) for bc in chain(self.bcu, self.bcσ)]
        import numpy as np
        print(np.linalg.matrix_rank(A.array()), np.linalg.norm(A.array()))
        it = solve(A, self.s_.vector(), b, 'bicgstab', 'hypre_amg')
        end()

        return it

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This makes the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.
        """
        self.s_n.assign(self.s_)
