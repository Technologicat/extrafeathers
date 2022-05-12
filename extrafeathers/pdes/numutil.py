# -*- coding: utf-8; -*-
"""Mathematically flavored utilities."""

__all__ = ["ε", "vol",
           "mag",
           "advw", "advs"]

from dolfin import sym, nabla_grad, dot, div, Identity, tr, dx, ds

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
    a UFL **form**; this generates both interior and boundary terms.

    Another way to view the role of the extra term in the skew-symmetric form is to
    consider the Helmholtz decomposition of the advection velocity `a`:

      a = ∇φ + ∇×A

    where `φ` is a scalar potential (for the irrotational part) and `A` is a
    vector potential (for the divergence-free part). We have

      (∇·a) = ∇·∇φ + ∇·∇×A = ∇²φ + 0

    so the extra term is proportional to the laplacian of the scalar potential:

      (∇·a) u = (∇²φ) u


    If `mode="general"`, i.e. `div(a) ≢ 0` is allowed, this routine uses the form

        [(a·∇) u + (1/2) (∇·a) u] - (1/2) (∇·a) u

    The integration by parts absorbs the first `(1/2) (∇·a) u`; the second one is
    converted to weak form simply as `∫ (1/2) (∇·a) u v dx`. Thus, in general mode,
    an extra symmetric term is produced.


    **References**:

        Jean Donea and Antonio Huerta. 2003. Finite Element Methods
        for Flow Problems. Wiley. ISBN 0-471-49666-9.
    """
    if mode not in ("general", "divergence-free"):
        raise ValueError(f"Expected `mode` to be one of 'general', 'divergence-free'; got {mode}")

    if mode == "general":
        # This is the weak form of `[(a·∇) u + (1/2) (∇·a) u] - (1/2) (∇·a) u`,
        # where the first `(1/2) (∇·a) u` is absorbed by the integration by parts.
        return (1 / 2) * ((dot(dot(a, nabla_grad(u)), v) -
                           dot(dot(a, nabla_grad(v)), u)) * dx +
                          dot(n, a) * dot(u, v) * ds -
                          div(a) * u * v * dx)  # the second (1/2) (∇·a) u
    else:  # mode == "divergence-free":
        # This is the skew-symmetric weak form of `(a·∇) u + (1/2) (∇·a) u`,
        # intended to be used when `div(a) ≡ 0`.
        return (1 / 2) * ((dot(dot(a, nabla_grad(u)), v) -
                           dot(dot(a, nabla_grad(v)), u)) * dx +
                           dot(n, a) * dot(u, v) * ds)

def advs(a, u, *, mode="divergence-free"):
    """Advection operator, strong form (for SUPG residual).

    Corresponds to the weak form produced by `advw`, which see.

    `a`: advection velocity
    `u`: quantity being advected
         `a` and `u` must be at least C0-continuous.
    `mode`: like `mode` of `advw`.

    Return value is an UFL expression representing the advection term.
    """
    if mode not in ("general", "divergence-free"):
        raise ValueError(f"Expected `mode` to be one of 'general', 'divergence-free'; got {mode}")

    if mode == "general":
        # Here the modifications cancel in the strong form;
        # we both add and subtract `(1/2) (∇·a) u`.
        return dot(a, nabla_grad(u))
    else:  # mode == "divergence-free":
        # To match `advw`, we must include the `(1/2) (∇·a) u` term that was added,
        # in case `a` is not actually exactly divergence-free.
        #
        # `advs` is most often used for computing the residual in SUPG
        # stabilization, so it needs to see the numerical error caused
        # by any nonzero values in the field `div(a)`.
        return dot(a, nabla_grad(u)) + (1 / 2) * div(a) * u
