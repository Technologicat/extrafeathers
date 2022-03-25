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

import typing

from fenics import (VectorFunctionSpace, TensorFunctionSpace,
                    TrialFunction, TestFunction,
                    Constant, Expression, Function,
                    FacetNormal, DirichletBC,
                    dot, inner, outer, sym, tr,
                    nabla_grad, div, dx, ds,
                    Identity,
                    lhs, rhs, assemble, solve,
                    interpolate, project, VectorSpaceBasis, as_backend_type,
                    errornorm,
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

    For now this is linear elastic or Kelvin-Voigt (chosen by hard-coding),
    but the plan is to extend this to SLS (the standard linear solid),
    and to make the constitutive law choosable.

    Time integration is performed using the θ method; Crank-Nicolson by default.

    `V`: function space for displacement
    `Q`: function space for stress
    `ρ`: density [kg / m³]
    `λ`: Lamé's first parameter [Pa]
    `μ`: shear modulus [Pa]
    `V0`: velocity of co-moving frame in +x direction (constant) [m/s]
    `bcu`: Dirichlet boundary conditions for displacement
    `bcv`: Dirichlet boundary conditions for Eulerian displacement rate ∂u/∂t
           (set these consistently with `u`)
    `bcσ`: Dirichlet boundary conditions for stress (set only n·σ).
           Alternative for setting `u` and `v`.
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
                 bcv: typing.List[DirichletBC],
                 bcσ: typing.List[DirichletBC],
                 dt: float, θ: float = 0.5):
        self.mesh = V.mesh()
        if Q.mesh() is not V.mesh():
            raise ValueError("V and Q must be defined on the same mesh.")

        # Trial and test functions
        #
        # `u`: displacement
        # `v`: Eulerian time rate of displacement (reduction to 1st order system)
        # `σ`: stress
        #  - Mixed formulation; stress has its own equation to allow easily
        #    changing the constitutive model.
        #  - Also, need to treat it this way for an Eulerian description of
        #    viscoelastic models, because the material derivative introduces
        #    a term that is one order of ∇ higher. In the primal formulation,
        #    in the weak form, this requires taking second derivatives of `u`.

        # # We won't use it, but this is how to set up the quantities for a monolithic
        # # mixed problem (in this example, for `u` and `σ`).
        # #
        # # Using a `MixedFunctionSpace` fails for some reason; instead, the way
        # # to do this is to set up a `MixedElement` and a garden-variety `FunctionSpace`
        # # on that, and then split as needed. Then set Dirichlet BCs on the appropriate
        # # `S.sub(j)` (those may also have their own second-level `.sub(k)` if they are
        # # vector/tensor fields).
        # #
        # e = MixedElement(V.ufl_element(), Q.ufl_element())
        # S = FunctionSpace(self.mesh, e)
        # u, σ = TrialFunctions(S)
        # w, φ = TestFunctions(S)
        # s_ = Function(S)
        # u_, σ_ = split(s_)

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
        self.VdG0 = VectorFunctionSpace(self.mesh, "DG", 0)
        self.QdG0 = TensorFunctionSpace(self.mesh, "DG", 0)

        self.u, self.v, self.σ = u, v, σ  # trials
        self.w, self.ψ, self.φ = w, ψ, φ  # tests
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
        #
        # The other equations have no space derivatives, just a projection of known data,
        # so no null space for them.

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

        null_space_basis = [interpolate(fu, V).vector() for fu in fus]

        # # In a mixed formulation, we must insert zero functions for the other fields:
        # zeroV = Function(V)
        # zeroV.vector()[:] = 0.0
        # zeroQ = Function(Q)
        # zeroQ.vector()[:] = 0.0
        # # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d5/dc7/classdolfin_1_1FunctionAssigner.html
        # assigner = FunctionAssigner(S, [V, Q])  # receiving space, assigning space
        # fss = [Function(S) for _ in range(len(fus))]
        # for fs, fu in zip(fss, fus):
        #     assigner.assign(fs, [project(fu, V), zeroQ])
        # null_space_basis = [fs.vector() for fs in fss]

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
        w = self.w
        u_ = self.u_    # latest available approximation
        u_n = self.u_n  # old (end of previous timestep)

        # Eulerian time rate of displacement,  v = ∂u/∂t
        v = self.v
        ψ = self.ψ      # test
        v_ = self.v_
        v_n = self.v_n

        # Stress
        σ = self.σ
        φ = self.φ
        σ_ = self.σ_
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
        # U = (1 - θ) * u_n + θ * u
        # V = (1 - θ) * v_n + θ * v
        # Σ = (1 - θ) * σ_n + θ * σ
        # dσdt = (σ - σ_n) / dt  # Kelvin-Voigt, SLS

        a = self.a  # convection velocity

        def advw(a, p, q):
            """Advection operator, weak form.

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected
            `q`: test function of `p`
            """
            return ((1 / 2) * (dot(dot(a, nabla_grad(p)), q) -
                               dot(dot(a, nabla_grad(q)), p)) * dx +
                               (1 / 2) * dot(n, a) * dot(p, q) * ds)
        def advs(a, p):
            """Advection operator, strong form (for SUPG residual).

            `a`: advection velocity (assumed divergence-free)
            `p`: quantity being advected
            """
            return dot(a, nabla_grad(p)) + (1 / 2) * div(a) * p

        # TODO:
        #  - Implement steady-state version (no `v` field needed)

        # Define variational problem

        # Step 1: ∂u/∂t = v -> obtain `u` (explicit in `v`)
        dudt = (u - u_n) / dt
        # V = v_n
        V = (1 - θ) * v_n + θ * v_  # known; initially `v_ = v_n` so just `v_n`, but this works iteratively too
        F_u = dot(dudt - V, w) * dx

        # Step 2: σ (using `u` from step 1)
        #
        # TODO:
        #  - Add elastothermal effects:  ∫ φ : [KE : α] [T - T0] dΩ  (same sign as ∫ φ : KE : ε dΩ term)
        #    - Need a FEM field for temperature T, and parameters α and T0
        #  - Add viscothermal effects, see eq. (768) in report
        #  - Orthotropic linear elastic
        #  - Orthotropic Kelvin-Voigt
        #  - Isotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #  - Orthotropic SLS (Zener), requires solving a PDE (LHS includes dσ/dt = ∂σ/∂t + (a·∇)σ)
        #
        # Linear elastic:
        #   σ = K : ε
        #   K = 2 μ ES + 3 λ EV
        # Because for any rank-2 tensor T,
        #   ES : T = symm(T)
        #   EV : T = vol(T) = (1/3) I tr(T)
        # we have
        #   σ = 2 μ symm(ε) + 3 λ vol(ε)
        #     = 2 μ ε + 3 λ vol(ε)
        #     = 2 μ ε + λ I tr(ε)
        # where on the second line we have used the symmetry of ε.

        # U = u_
        # V = v_
        U = (1 - θ) * u_n + θ * u_  # known
        V = (1 - θ) * v_n + θ * v_  # known
        εu = ε(U)
        εv = ε(V)
        # Σ = σ
        Σ = (1 - θ) * σ_n + θ * σ   # unknown

        # Linear elastic
        # stress_expr = 2 * μ * εu + λ * Identity(εu.geometric_dimension()) * tr(εu)

        # Axially moving Kelvin-Voigt
        τ_ret = 0.1  # Kelvin-Voigt retardation time (τ_ret := η/E)  TODO: parameterize
        # σ = 2 [μ + μ_visc d/dt] ε + I tr([λ + λ_visc d/dt] ε)
        #   = 2 μ [1 + τ_ret d/dt] ε + λ I tr([1 + τ_ret d/dt] ε)
        #   = 2 μ [1 + τ_ret (∂/∂t + a·∇)] ε + λ I tr([1 + τ_ret (∂/∂t + a·∇)] ε)
        stress_expr = (2 * μ * (εu + τ_ret * (εv + advs(a, εu))) +
                       λ * Identity(εu.geometric_dimension()) * (tr(εu) + τ_ret * (tr(εv) + advs(a, tr(εu)))))
        F_σ = inner(Σ - stress_expr, φ) * dx

        # # alternative: delta formulation (but needs some care when applying BCs)
        # Δu = u_ - u_n  # known
        # εΔu = ε(Δu)
        # Δσ = σ - σ_n  # unknown
        # Δstress_expr = 2 * μ * εΔu + λ * Identity(εΔu.geometric_dimension()) * tr(εΔu)
        # F_σ = inner(Δσ - Δstress_expr, φ) * dx

        # Step 3: v (momentum equation)
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
        dvdt = (v - v_n) / dt
        # U = (1 - θ) * u_n + θ * u_  # known
        # V = (1 - θ) * v_n + θ * v   # unknown!
        Σ = (1 - θ) * σ_n + θ * σ_  # known
        # Σ = σ_
        F_v = (ρ * dot(dvdt, ψ) * dx +
               2 * ρ * advw(a, V, ψ) -
               ρ * dot(dot(a, nabla_grad(U)), dot(a, nabla_grad(ψ))) * dx +
               inner(Σ.T, ε(ψ)) * dx -
               dot(dot(n, Σ.T), ψ) * ds +
               ρ * dot(n, dot(dot(outer(a, a), nabla_grad(U)), ψ)) * ds -  # +∫ ρ ([a⊗a]·∇u)·ψ dx
               ρ * dot(b, ψ) * dx)
        # # DEBUG
        # F_v = (ρ * dot(dvdt, ψ) * dx +
        #        inner(Σ.T, ε(ψ)) * dx -
        #        dot(dot(n, Σ.T), ψ) * ds -
        #        ρ * dot(b, ψ) * dx)

        # SUPG: streamline upwinding Petrov-Galerkin.
        def mag(vec):
            return dot(vec, vec)**(1 / 2)
        τ_SUPG = α0 * (1 / (θ * dt) + 2 * mag(a) / he + 4 * (μ / ρ) / he**2)**-1  # [τ] = s  # TODO: tune value
        # The residual is evaluated elementwise in strong form,
        # at the end of the timestep.
        R = (ρ * ((v - v_n) / dt + 2 * advs(a, v) + advs(a, advs(a, u_))) -
             div(σ_) - ρ * b)
        F_SUPG = enable_SUPG_flag * τ_SUPG * dot(advs(a, ψ), R) * dx
        F_v += F_SUPG

        self.a_u = lhs(F_u)
        self.L_u = rhs(F_u)
        self.a_σ = lhs(F_σ)
        self.L_σ = rhs(F_σ)
        self.a_v = lhs(F_v)
        self.L_v = rhs(F_v)

    def step(self) -> typing.Tuple[int, int, int, typing.Tuple[int, float]]:
        """Take a timestep of length `self.dt`.

        Updates the latest computed solution.
        """
        begin("Solve timestep")

        v_prev = Function(self.V)
        maxit = 100  # TODO: parameterize
        tol = 1e-8  # TODO: parameterize
        for _ in range(maxit):
            v_prev.assign(self.v_)

            # Step 1: update `u`
            A1 = assemble(self.a_u)
            b1 = assemble(self.L_u)
            [bc.apply(A1) for bc in self.bcu]
            [bc.apply(b1) for bc in self.bcu]
            it1 = solve(A1, self.u_.vector(), b1, 'cg', 'sor')

            # Postprocess `u` to eliminate numerical oscillations
            self.u_.assign(project(interpolate(self.u_, self.VdG0), self.V))

            # Step 2: update `σ`
            A2 = assemble(self.a_σ)
            b2 = assemble(self.L_σ)
            [bc.apply(A2) for bc in self.bcσ]
            [bc.apply(b2) for bc in self.bcσ]
            # Axially moving Kelvin-Voigt needs a non-symmetric solver here due to the a·∇ε term.
            # (We use the skew-symmetric discretization, which is often better than the naive one,
            #  but it's still non-symmetric.)
            it2 = solve(A2, self.σ_.vector(), b2, 'bicgstab', 'sor')

            # Postprocess `σ` to eliminate numerical oscillations
            self.σ_.assign(project(interpolate(self.σ_, self.QdG0), self.Q))

            # Step 3: tonight's main event (solve momentum equation for `v`)
            A3 = assemble(self.a_v)
            b3 = assemble(self.L_v)

            if self.bcv:
                [bc.apply(A3) for bc in self.bcv]
                [bc.apply(b3) for bc in self.bcv]
            else:
                # Eliminate rigid-body motion solutions of momentum equation (for Krylov solvers)
                #
                # `set_near_nullspace`: "Attach near nullspace to matrix (used by preconditioners,
                #                        such as smoothed aggregation algebraic multigrid)"
                # `set_nullspace`:      "Attach nullspace to matrix (typically used by Krylov solvers
                #                        when solving singular systems)"
                #
                # https://fenicsproject.org/olddocs/dolfin/latest/cpp/d4/db0/classdolfin_1_1PETScMatrix.html#aeb0152c4382d473ae6a93841f721260c
                #
                A3_PETSc = as_backend_type(A3)
                A3_PETSc.set_near_nullspace(self.null_space)
                A3_PETSc.set_nullspace(self.null_space)
                # self.null_space.orthogonalize(b3)  # TODO: what goes wrong here?

            it3 = solve(A3, self.v_.vector(), b3, 'bicgstab', 'hypre_amg')

            # Postprocess `v` to eliminate numerical oscillations
            self.v_.assign(project(interpolate(self.v_, self.VdG0), self.V))

            e = errornorm(self.v_, v_prev, 'h1', 0, self.mesh)
            if e < tol:
                break

        # # DEBUG
        # import numpy as np
        # print(np.linalg.matrix_rank(A.array()), np.linalg.norm(A.array()))
        # print(sum(np.array(b) != 0.0), np.linalg.norm(np.array(b)), np.array(b))

        end()

        return it1, it2, it3, ((_ + 1), e)

    def commit(self) -> None:
        """Commit the latest computed timestep, preparing for the next one.

        This makes the latest computed solution the "old" solution for
        the next timestep. The old "old" solution is discarded.
        """
        self.u_n.assign(self.u_)
        self.v_n.assign(self.v_)
        self.σ_n.assign(self.σ_)
