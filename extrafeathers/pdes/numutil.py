# -*- coding: utf-8; -*-
"""Mathematically flavored utilities."""

__all__ = ["Min", "Max", "Minn", "Maxx",
           "ε", "vol",
           "mag",
           "advw", "advs"]

from dolfin import sym, nabla_grad, dot, div, Identity, tr, dx, ds, Constant

def Min(a, b):
    """UFL expression for the min of expressions `a` and `b`.

    Based on:
        https://fenicsproject.org/qa/10199/using-min-max-in-a-variational-form/
    """
    return (a + b - abs(a - b)) / Constant(2)

def Max(a, b):
    """UFL expression for the max of expressions `a` and `b`.

    Based on:
        https://fenicsproject.org/qa/10199/using-min-max-in-a-variational-form/
    """
    return (a + b + abs(a - b)) / Constant(2)

def Minn(*terms):
    """UFL expression for the min of expressions in `terms`.

    Implemented recursively in terms of `Min`, which see.
    """
    a, *rest = terms
    if len(rest) == 1:
        return Min(a, rest[0])
    return Min(a, Minn(*rest))

def Maxx(*terms):
    """UFL expression for the max of expressions in `terms`.

    Implemented recursively in terms of `Max`, which see.
    """
    a, *rest = terms
    if len(rest) == 1:
        return Max(a, rest[0])
    return Max(a, Maxx(*rest))

def ε(u):
    """UFL expression for the symmetric gradient of the field `u`.

        (symm ∇)(u) = (1/2) (∇u + transpose(∇u))

    For example, to plot the 11 component of `symm ∇u`::

        from fenics import interpolate, plot

        # scalar function space
        W = V.sub(0).collapse()  # `V` is a `VectorFunctionSpace`
        # W = FunctionSpace(mesh, 'P', 2)  # can do this, too

        εu = ε(u)  # `u` is a `Function` on `V`
        εu11 = εu.sub(0)
        plot(interpolate(εu11, W))

    Note in FEniCS the tensor components are indexed linearly,
    and the storage is by row. E.g. in 2D::

        εu11 = εu.sub(0)
        εu12 = εu.sub(1)
        εu21 = εu.sub(2)
        εu22 = εu.sub(3)

    See:
        https://fenicsproject.org/qa/4458/how-can-i-get-two-components-of-a-tensorfunction/
    """
    return sym(nabla_grad(u))

def vol(T):
    """Volumetric part of rank-2 tensor `T`."""
    d = T.geometric_dimension()
    Id = Identity(d)
    return 1 / d * Id * tr(T)

def mag(vec):
    """UFL expression for the magnitude of vector `vec`."""
    return dot(vec, vec)**(1 / 2)

def advw(a, u, v, n, *, mode="divergence-free"):
    """Advection operator, skew-symmetric weak form.

    The skew-symmetric form typically improves numerical stability,
    especially for divergence-free advection velocity fields.

    `a`: advection velocity vector field
    `u`: quantity being advected
    `v`: test function of the quantity `u`
         `u` and `v` must be at least C0-continuous.
         `u` and `v` must be the same kind; scalar and vector are supported.
         When `mode="general"`, `a` must be at least C0-continuous.
    `n`: facet normal of mesh

    `mode`: one of "general" or "divergence-free".

            If "divergence-free", it is assumed that `div(a) ≡ 0`.
            The form produced is only valid for divergence-free `a`.

            If "general", the form produced is valid for any `a`,
            but beside the skew-symmetric part, includes also a
            symmetric term.

    Return value is an UFL form representing the advection term.


    **Background**:

    Donea & Huerta (2003, sec. 6.7.1) remark that in the 2000s, it has
    become standard to discretize the advection operator in Navier-Stokes
    in this skew-symmetric weak form:

      (1/2) a · [∇u · v - ∇v · u] dx

    which in the strong form is equivalent with replacing the advection term

      (a·∇) u

    by the modified term

      (a·∇) u  +  (1/2) (∇·a) u

    This is consistent when `div(a) ≡ 0`, and in Navier-Stokes, necessary for
    unconditional time stability for schemes that are able to provide it.

    Zang (1991) hints that this is due to better conservation properties
    of the skew-symmetric form (momentum, kinetic energy).

    To see the equivalence, consider the conversion of the modified term
    into weak form:

       ((a·∇) u) · v dx  +  (1/2) (∇·a) u · v dx

    Observing that

       ∂i (ai uk vk) = (∂i ai) uk vk + ai ∂i (uk vk)
       ∇·(a (u · v)) = (∇·a) u · v  +  a · ∇(u · v)

    we use the divergence theorem in the last term of the weak form, obtaining

       (a·∇) u · v dx  -  (1/2) a · ∇(u · v) dx  +  (1/2) n · a (u · v) ds

    Furthermore, noting that

       a · ∇(u · v) = ai ∂i (uk vk)
                     = ai (∂i uk) vk + ai uk (∂i vk)
                     = a · ∇u · v  +  a · ∇v · u
                     = a · [∇u · v + ∇v · u]

    and

       ((a·∇) u) · v = ((ai ∂i) uk) vk = ai (∂i uk) vk = a · ∇u · v

    we have the terms

         a · ∇u · v dx
       - (1/2) a · [∇u · v + ∇v · u] dx
       + (1/2) n · a (u · v) ds

    Cleaning up, we obtain

       (1/2) a · [∇u · v - ∇v · u] dx  +  (1/2) n · a (u · v) ds

    as claimed. Keep in mind the boundary term, which contributes on boundaries
    through which there is flow (i.e. inlets and outlets) - we do not want to
    introduce an extra boundary condition. This is why this routine returns
    a UFL **form**; the skew-symmetric discretization of the advection operator
    generates both interior and boundary terms.

    If `mode="general"`, i.e. `div(a) ≢ 0` is allowed, this routine uses the form

        [(a·∇) u + (1/2) (∇·a) u] - (1/2) (∇·a) u

    The integration by parts absorbs the first `(1/2) (∇·a) u`, as above; the second
    one is kept as `-∫ (1/2) (∇·a) u · v dx`. Thus, when `mode="general"`, an extra
    symmetric term is produced.


    **Remark 1**

    Why does skew-symmetric advection help stability for incompressible flow simulations?
    From https://en.wikipedia.org/wiki/Advection :

        Since skew symmetry implies only imaginary eigenvalues, this form reduces the "blow up"
        and "spectral blocking" often experienced in numerical solutions with sharp discontinuities
        (see Boyd, 2000).


    **Remark 2**

    Role of the extra terms? Let's consider `mode="general"`, and flip the sign for clarity.

      - For any test function `v` with support on a part of ∂Ω, the extra terms are:

          ∫ [∇·a]u · v dx - ∫ [n·a]u · v ds

        This represents a continuum balance of how much `u` enters the support of `v`:
        the source of `u` due to [∇·a], minus the outward flow of `u` through ∂Ω.
        (Sign: if ∇·a > 0 at a point, the velocity field flows outward from that point.)

      - For any test function `v` whose support is contained in the interior of Ω,
        we have just one extra term:

          ∫ [∇·a]u · v dx

        which describes the source of `u` due to [∇·a].

    Thus, what this technique essentially does is that it splits the advection operator
    into a skew-symmetric divergence-free (incompressible) transport operator, plus a
    symmetric operator that accounts for sources/sinks.


    **Remark 3**

    Another way to view the role of the extra term is to consider the Helmholtz
    decomposition of the advection velocity `a`:

      a = ∇φ + ∇×A

    where `φ` is a scalar potential (for the irrotational part) and `A` is a
    vector potential (for the divergence-free part). We have

      (∇·a) = ∇·∇φ + ∇·∇×A = ∇²φ + 0

    so the extra term is proportional to the laplacian of the scalar potential:

      (∇·a) u = (∇²φ) u

    This obviously hints at the same conclusion as above.


    **Remark 4**

    For incompressible Navier-Stokes, Zang (1991) writes the skew-symmetric form as:

        (1/2) u·∇u + (1/2) ∇·(uu)

    The terms  u·∇u  and  ∇·(uu)  are known, respectively, as the /convection/ and
    /divergence/ forms of the Navier-Stokes advection term. The skew-symmetric form
    is their arithmetic mean. There is a large literature on these various forms
    as applied to the incompressible Navier-Stokes equations, particularly in the
    context of finite difference and finite volume discretizations.

    In our notation, the divergence form is

          ∇·(u ⊗ u)
        ≡ ∂i (ui uk)
        = (∂i ui) uk + ui (∂i uk)
        ≡ (∇·u) u + u·∇u

    or for a general velocity field `a`:

          ∇·(a ⊗ u)
        ≡ ∂i (ai uk)
        = (∂i ai) uk + ai (∂i uk)
        ≡ (∇·a) u + a·∇u

    which is exactly what we had in the arguments above.


    **References**:

        Jean Donea and Antonio Huerta. 2003. Finite Element Methods
        for Flow Problems. Wiley. ISBN 0-471-49666-9.

        Thomas Zang. 1991. On the rotation and skew-symmetric forms for incompressible flow simulations.
        Applied Numerical Mathematics. 7: 27–40. doi:10.1016/0168-9274(91)90102-6

        John P. Boyd. 2000. Chebyshev and Fourier Spectral Methods 2nd edition. Dover. p. 213.
    """
    if mode not in ("general", "divergence-free"):
        raise ValueError(f"Expected `mode` to be one of 'general', 'divergence-free'; got {mode}")

    if mode == "general":
        # This is the weak form of `[(a·∇) u + (1/2) (∇·a) u] - (1/2) (∇·a) u`,
        # where the first `(1/2) (∇·a) u` is absorbed by the integration by parts,
        # producing the boundary term. The second `(1/2) (∇·a) u` is kept as-is.
        return (1 / 2) * ((dot(dot(a, nabla_grad(u)), v) -
                           dot(dot(a, nabla_grad(v)), u)) * dx +
                          dot(n, a) * dot(u, v) * ds -
                          div(a) * dot(u, v) * dx)  # the second (1/2) (∇·a) u
    else:  # mode == "divergence-free":
        # This is the skew-symmetric weak form of `(a·∇) u + (1/2) (∇·a) u`,
        # intended to be used when `div(a) ≡ 0`. Note that for consistency,
        # we must still account for the flow of `u` through ∂Ω.
        return (1 / 2) * ((dot(dot(a, nabla_grad(u)), v) -
                           dot(dot(a, nabla_grad(v)), u)) * dx +
                           dot(n, a) * dot(u, v) * ds)

def advs(a, u, *, mode="divergence-free"):
    """Advection operator, strong form.

    Useful in computing the strong-form residual for SUPG stabilization.

    The result is consistent with the weak form produced by `advw` with the
    same `mode` setting, which see.

    `a`: advection velocity
    `u`: quantity being advected
         `a` and `u` must be at least C0-continuous.
         `u` must be scalar or vector.
    `mode`: like `mode` of `advw`.

    Return value is an UFL expression representing the advection term.
    """
    if mode not in ("general", "divergence-free"):
        raise ValueError(f"Expected `mode` to be one of 'general', 'divergence-free'; got {mode}")

    if mode == "general":
        # In this mode, our modifications cancel; we both add and subtract `(1/2) (∇·a) u`.
        # Thus we are left with just the original convection form.
        return dot(a, nabla_grad(u))
    else:  # mode == "divergence-free":
        # To make the result consistent with the weak form returned by `advw` in the same mode,
        # we must include the extra `(1/2) (∇·a) u` term - which we initially add, but then absorb
        # into the integration by parts.
        #
        # This is because in practice, `a` might be only approximately divergence-free
        # (such as a velocity field produced by an incompressible Navier-Stokes solver).
        return dot(a, nabla_grad(u)) + (1 / 2) * div(a) * u
