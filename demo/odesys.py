#!/usr/bin/env python
# -*- coding: utf-8; -*-
"""
How to repurpose FEniCS as an ODE system solver:
https://fenicsproject.discourse.group/t/can-one-solve-system-of-ode-with-fenics/486
"""

from dolfin import (IntervalMesh, interval,
                    MixedElement, split, FiniteElement, FunctionSpace,
                    DirichletBC, Constant, Expression, Function,
                    TestFunction, TrialFunction,
                    inner, dx,
                    grad, Dx, derivative,
                    lhs, rhs, assemble, solve,
                    SpatialCoordinate,
                    sqrt, sin, cos, pi,
                    project,
                    plot)

def main(ncells):
    mesh = IntervalMesh(ncells, 0, 2 * pi)

    Welm = MixedElement([FiniteElement('Lagrange', interval, 2),
                         FiniteElement('Lagrange', interval, 1)])
    W = FunctionSpace(mesh, Welm)

    bcsys = [DirichletBC(W.sub(0), Constant(0.0), 'near(x[0], 0)'),
             DirichletBC(W.sub(1), Constant(1.0), 'near(x[0], 0)')]

    # Original problem:
    #
    #   u'' = u - f(x)
    #   u(0) = 0
    #   u'(0) = 1
    #
    # where
    #
    #   f(x) := 2 sin(x)
    #
    # Splitting by introducing the auxiliary variable p = u':
    #
    #   u' = p
    #   p' = u - f(x)
    #   u(0) = 0
    #   p(0) = 1
    #
    u, p = split(TrialFunction(W))
    v, q = split(TestFunction(W))
    f = Expression('2.0*sin(x[0])', degree=2)

    # Weak forms:
    #
    #   ∫ u' v dx - ∫ p v dx = 0
    #   ∫ p' q dx - ∫ u q dx + ∫ f q dx = 0
    #
    # v and q are arbitrary and independent, so we may collect this into one equation:
    #
    #   ∫ u' v dx - ∫ p v dx + ∫ p' q dx - ∫ u q dx + ∫ f q dx = 0
    #
    weak_form = u.dx(0) * v * dx - p * v * dx + p.dx(0) * q * dx - u * q * dx + f * q * dx
    # weak_form = grad(u)[0] * v * dx - p * v * dx + grad(p)[0] * q * dx - u * q * dx + f * q * dx
    # weak_form = Dx(u, 0) * v * dx - p * v * dx + Dx(p, 0) * q * dx - u * q * dx + f * q * dx

    # solve as nonlinear problem (since it's linear, the Newton algorithm will converge in one iteration)
    # solve(weak_form == 0, up, bcs=bcsys)

    # Jac = derivative(weak_form, up, TrialFunction(W))
    # solve(weak_form == 0, up, J=Jac, bcs=bcsys)

    # # extract lhs/rhs, solve as linear problem
    a = lhs(weak_form)
    L = rhs(weak_form)

    up = Function(W, name="u_and_p")  # subfields can't be named separately; they will be automatically named "u_and_p-0", "u_and_p-1"
    solve(a == L, up, bcs=bcsys)

    u, p = split(up)
    x, = SpatialCoordinate(mesh)

    print('ncells =', ncells)
    print('u L2 error', sqrt(abs(assemble(inner(u - sin(x), u - sin(x)) * dx))))  # compare to known solution
    print('p L2 error', sqrt(abs(assemble(inner(p - cos(x), p - cos(x)) * dx))))

    # https://fenicsproject.org/qa/11979/significance-of-collapsing/
    Wu = W.sub(0).collapse()
    plot(project(u, Wu))

list(map(main, [10**i for i in range(1, 5)]))
import matplotlib.pyplot as plt
plt.show()
