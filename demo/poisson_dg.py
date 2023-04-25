#!/usr/bin/env python
# -*- coding: utf-8; -*-
"""Small experiment with the Symmetric interior penalty dG method (SIPG).

Based on the FEniCS Poisson demos (both documented and undocumented),
and various internet sources. Links provided in source code comments.

For now, this is mainly here to document how to do this; a dG Poisson solver
could be added to `extrafeathers.pdes` later.
"""

import numpy as np
import matplotlib.pyplot as plt

from dolfin import (UnitSquareMesh, FunctionSpace, DOLFIN_EPS, Constant,  # DirichletBC,
                    TrialFunction, TestFunction, Expression,
                    inner, grad, dot, avg, jump, dS, dx, ds, FacetNormal,
                    Function, lhs, rhs, assemble, solve, XDMFFile,  # Point,
                    MPI, parameters)
# from mshr import Rectangle, Circle, generate_mesh

import extrafeathers


# needed when `dS` is used in parallel; set the mode before instantiating the mesh
parameters["ghost_mode"] = "shared_vertex"

N = 8
mesh = UnitSquareMesh(N, N)

# # Create mesh (L-shaped domain)
# bigbox = Rectangle(Point(0, 0), Point(1, 1))
# smallbox = Rectangle(Point(0, 0), Point(0.5, 0.5))
# domain = bigbox - smallbox
# # circle = Circle(Point(0.4, 0.4), 0.3)
# # domain = bigbox - circle
# mesh = generate_mesh(domain, N)

# Note dG0 won't work, because:
#  - We need to take the gradient, and for dG0 it's identically zero.
#  - dG0 has no DOFs on element edges, so it does not see the facet integrals.
V = FunctionSpace(mesh, "DG", 2)
# V = FunctionSpace(mesh, "P", 1)  # for classical Galerkin
u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(mesh)

# Dirichlet boundary
u0 = Constant(1)
def boundary(x):
    return (x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or
            x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS)
# bc = DirichletBC(V, u0, boundary)  # we'll enforce this weakly, with the Nitsche trick

# Load functions
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
# g = Expression("sin(5*x[0])", degree=2)  # Neumann BC load

# Variational problem.
#
# Let's begin with some hopefully useful notes:
#
# https://fenicsproject.org/pub/course/lectures/2017-nordic-phdcourse/lecture_10_discontinuous_galerkin.pdf
#
#     We want to weakly enforce
#     • Continuity of the flux: [∇u] = 0 over all facets
#     (Solution: Let the corresponding term vanish)
#     • Continuity of the solution [u] = 0 over all facets
#     (Solution: Add a corresponding term)
#     • Stability
#     (Solution: add:
#        S(u, v) = ∑[e∈F] ∫e (β/h) [u] · [v] ds
#     for some stabilization parameter β > 0 and mesh size h.)
#
# The jump identity is
#   [U v] = [U] <v> + <U> · [v]
# where U is a vector, v is a scalar, [] is the jump and <> is the average:
#   [U] = (U+ - U-) · n  (the jump of a vector is a scalar)
#   [v] = (v+ - v-) n    (the jump of a scalar is a vector)
#   <U> = (1/2) (U+ + U-)
#   <v> = (1/2) (v+ + v-)
# and `n` points from the `+` side to the `-` side.
#
# Here integration by parts produces, on the interior facets, the term:
#   n·(∇u+ v+ - ∇u- v-) = [∇u v] = [∇u] <v> + <∇u> · [v]
# Letting [∇u] → 0 in the result (weakly enforcing the continuity of the flux),
#   n·(∇u+ v+ - ∇u- v-) = <∇u> · [v]
# which is the term labeled as "integration by parts" below.
#
# https://fenicsproject.org/qa/9020/derivation-of-weak-form-in-undocumented-dg-poisson-demo/
#
# Christian Waluga writes:
#     You can always add consistent terms, since in the derivation they are zero
#     for exact solutions of your problem (u can be seen as continuous then,
#     sufficient regularity assumed). They only matter in a discrete sense when you
#     want to show stability of the finite-dimensional problem. Since u = u0 on the
#     [Dirichlet] boundary, you can also add terms like (u - u0) times something,
#     it doesn't matter in the derivation. Where it matters is in the discrete
#     sense, since they penalize the jump at the boundaries, thus weakly
#     enforce Dirichlet conditions for sufficiently large gamma (something like
#     10 times the polynomial degree squared is often sufficient). The terms
#     with alpha are only present on interior facets in the example code (dS is
#     for interior, ds for exterior facets, a small but important difference in
#     the caps).

# maartent writes:
#     In the derivation as I have it now, I assume that the continuity of the solution,
#     [u] = 0, is something that only has meaning in the interior facets, so the correct
#     term to add for symmetry is ∫Γi ⟨∇v⟩ [u] dS for the interior facets only, and then
#     another ∫ΓD (u − u0) ∇v·n ds for the Dirichlet boundary, as you tried to tell me.
#
# See also:
# https://scicomp.stackexchange.com/questions/20078/matlab-implementation-of-2d-interior-penalty-discontinuous-galerkin-poisson-prob
# https://fenicsproject.org/qa/3974/expression-in-interior-facets/
#
#
# Also see Brenner & Scott, sec. 10.5. Reading between the lines, the motivation of the
# stabilization term seems to be that it is needed to set up a suitable energy norm
# from which one can prove coercivity, whence stability.
#
# The authors also remark (p. 291) that for a consistent method, convergence only
# depends on stability, which in turn involves boundedness (i.e. continuity in the
# bilinear form sense) and coercivity of the relevant bilinear forms. It turns out that
# there is a critical value for the stabilization parameter; the parameter must be
# critical or larger for the discrete bilinear form to be coercive. See lemma 10.5.19.
#
# Reference:
#   Susanne C. Brenner & L. Ridgway Scott. 2010. The Mathematical Theory
#   of Finite Element Methods. 3rd edition. Springer. ISBN 978-1-4419-2611-1.
#
#
# TL;DR summary: in a consistent formulation, we can arbitrarily add:
#   - Terms with [u] on the interior facets
#   - Terms with (u - u0) on the Dirichlet boundary facets
# because these vanish for the exact solution `u`.
#
# On the external boundary `ds`, the added terms must be applied only on the
# Dirichlet boundaries (which are here enforced weakly, with the Nitsche trick).
# If we have any Neumann or Robin BCs, we must set up subdomains for `ds`.
# We must split `ds` to apply BCs selectively by boundary tag, and
# include a list of boundary tags in the Neumann BC specification.
# This is not done in this simple demo.
#
# See the tutorial:
#   https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html#fenics-implementation-14
# particularly, how to redefine the measure `ds` in terms of boundary markers:
#
#   ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundary_parts)
#
# Then use `ds(i)` as the integration symbol, where `i` is the boundary number
# in the mesh data, e.g. `Boundaries.LEFT.value` from the problem configuration.
#
γ = Constant(10 * V.ufl_element().degree()**2)  # Nitsche parameter
α = Constant(10 * γ)  # stabilization parameter
he = extrafeathers.cell_mf_to_expression(extrafeathers.meshsize(mesh))
F = (inner(grad(u), grad(v)) * dx +
     (-dot(avg(grad(u)), jump(v, n)) +  # integration by parts, after split to triangles
      -dot(avg(grad(v)), jump(u, n)) +  # ∫Γi ⟨∇v⟩ [u] dS, added for symmetry of LHS
       α / avg(he) * dot(jump(u, n), jump(v, n))) * dS +  # stabilization
     (-dot(grad(u), n) * v +
      -dot(grad(v), n) * (u - u0) +  # ∫ΓD (u − u0) ∇v·n ds, added for symmetry of LHS
      γ / he * (u - u0) * v) * ds +  # Nitsche Dirichlet BC
     -f * v * dx)  # - g * v * ds(Neumann)

# For comparison:

# # Classical continuous Galerkin
# F = inner(grad(u), grad(v)) * dx - f * v * dx  # - g * v * ds(Neumann)

# # Classical continuous Galerkin with the Nitsche trick
# F = (inner(grad(u), grad(v)) * dx +
#      γ / he * (u - u0) * v * ds +  # Nitsche Dirichlet BC
#      -f * v * dx)  # - g * v * ds(Neumann)

a = lhs(F)
L = rhs(F)

# # Compute solution
# u_ = Function(V)
# solve(a == L, u_, bc)

A = assemble(a)
b = assemble(L)

# Maybe useful for debugging (e.g. to see that using dG0 leads to the zero matrix):
# print(np.linalg.matrix_rank(A.array()), np.linalg.norm(A.array()), A.array())

# bc.apply(A)
# bc.apply(b)
u_ = Function(V)
it = solve(A, u_.vector(), b, 'cg', 'sor')
# it = solve(A, u_.vector(), b, 'bicgstab', 'hypre_amg')  # umm, no.
if MPI.comm_world.rank == 0:
    print(f"Solved in {it} CG iterations")

print(f'Process {MPI.comm_world.rank}: max(u) = {np.array(u_.vector()).max():0.6g}')

# vtkfile = File("demo/output/poisson_dg/solution.pvd")
# vtkfile << u_
# Create XDMF file for visualization output
xdmffile_u = XDMFFile(MPI.comm_world, 'demo/output/poisson_dg/u.xdmf')
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False
xdmffile_u.write(u_, 0)  # (field, time)

# --------------------------------------------------------------------------------
# Visualize

if MPI.comm_world.rank == 0:
    print("Plotting.")
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 8))
theplot = extrafeathers.mpiplot(u_, show_mesh=True, show_partitioning=True)
if MPI.comm_world.rank == 0:
    plt.colorbar(theplot)
    plt.legend(loc="upper right")  # show the labels for the mesh parts
    plt.axis("equal")
    plt.title(f"Poisson with dG({V.ufl_element().degree()}) + SIPG")
    plt.show()
