#!/usr/bin/env python
# -*- coding: utf-8; -*-
"""For the FEniCS-technical side, this is based on `demo/odesys.py`, which in turn is based on:
https://fenicsproject.discourse.group/t/can-one-solve-system-of-ode-with-fenics/486
"""

import numpy as np

from dolfin import (IntervalMesh, interval,
                    FiniteElement, FunctionSpace, CompiledSubDomain,
                    DirichletBC, Constant, Expression, Function,
                    TestFunction, TrialFunction,
                    dx,
                    lhs, rhs, assemble, solve,
                    errornorm,
                    project,
                    plot,
                    MPI, Vector)

from extrafeathers import meshmagic

from .config import T0, T_ext, Γ, R, c_func, rho

my_rank = MPI.comm_world.rank


def estimate(tmax=20.0, nel=2000, degree=2):
    """Compute an educated guesstimate for the temperature profile at the inlet edge.

    This automatically takes the material parameter config from `config.py`.
    Solution method is finite elements in time, using classical nodal Lagrange elements.
    If `tmax` is larger than a hardcoded limit, the solution proceeds piecewise.
    Once `u - u_ext` falls below a hardcoded tolerance, the simulation stops,
    and a final data point `(tmax, u_ext)` is appended to the result.

    We consider a small sphere of material, exposed to the environment, and integrate
    Newton's law of cooling. This gives us the temperature of the sphere as a function of time.

    By defining how long it takes in the printing process for the laser to return to the
    same spot (to print another layer), one can use this temperature profile to roughly
    construct a temperature profile as a function of the depth coordinate.

    (Obviously, we neglect the presence of neighboring material points; each point cools
    as if it was separately exposed to the environment.)

    Parameters:

        `tmax`: End time for this 0D cooling simulation
        `nel`: Number of elements. If solving piecewise, this is for each piece.
               The default `nel` has been chosen to work well with the default `tmax`
               when simulating 316L steel, starting from its solidus temperature
               at 1700 K, and exposed to air at room temperature (22°C, 295.15 K).
        `degree`: Degree of the Lagrange elements

    Return value is `(xx, uu)`, where:

        `xx`: rank-1 `np.array`, time coordinate of each node of the mesh
        `uu`: rank-1 `np.array`, solution value at the nodes

    We return only NumPy arrays (no FEniCS function), because for large `tmax`, the
    solution proceeds piecewise, so there is no global function for all times `t`.
    """
    if my_rank == 0:
        print(f"Solving inlet temperature profile with {nel} elements (degree {degree}), up to t = {tmax} s")

    # How does a sphere of radius R cool, starting from initial temperature `u0`,
    # and exposed to an environment at temperature `u_ext`? Quick pen-and-paper
    # derivation:
    #
    # Heat equation for a single material point (here `u` [K] is the temperature),
    # with Newton's law of cooling:
    #   ρ c u' = -ρ r [u - u_ext]
    #   u(0) = u0
    #
    # where
    #   r = dA/dm Γ,   [Γ] = W/m²
    #
    # The equation follows from the heat equation for a moving material, by requiring
    # that all fields are constant in space, and neglecting the mechanical work term.
    #
    # Consider a sphere with radius R and constant density ρ,
    #   A = 4 π R²
    #   m = ρ V
    # where
    #   V = 4/3 π R³
    #
    # For any fixed R, the exposed surface area per unit mass is:
    #   A / m = (4 π R²) / (ρ 4/3 π R³)
    #         = 3 / (ρ R)
    #
    # Divide the ODE by ρ c:
    #   u' = -(dA/dm Γ / c) [u - u_ext]
    #      = -(3 Γ / (R ρ c)) [u - u_ext]
    #
    # Note that  c = c(u).
    #
    # Weak form. Multiply by a scalar test function `v`, integrate over the domain
    # `(0, tmax)`, and move all terms to the left-hand side. We obtain:
    #
    #   ∫ u' v dx + (3 Γ / (R ρ)) ∫ (1 / c) [u - u_ext] v dx = 0
    #
    def solve_piece(t0, tmax, T0):
        # Remap the interval to start from `t = 0`. This avoids triggering recompilation
        # of the compiled subdomain for the BC, and thus runs much faster (especially
        # when doing a piecewise solve).
        tmax = tmax - t0

        mesh = IntervalMesh(nel, 0.0, tmax)

        Welm = FiniteElement("Lagrange", interval, degree)
        W = FunctionSpace(mesh, Welm)

        # t0_boundary = CompiledSubDomain(f"near(x[0], {t0})")
        # bcu = [DirichletBC(W, Constant(T0), t0_boundary)]
        # bcu = [DirichletBC(W, Constant(T0), f"near(x[0], {t0})")]
        bcu = [DirichletBC(W, Constant(T0), "near(x[0], 0.0)")]

        u = TrialFunction(W)
        v = TestFunction(W)

        c = Function(W, name="c")
        weak_form = (u.dx(0) * v * dx +
                     Constant(3 * Γ / (R * rho)) / c * (u - T_ext) * v * dx)
        a = lhs(weak_form)
        L = rhs(weak_form)

        u_ = Function(W, name="u")
        u_prev = Function(W, name="u_prev")

        # Nonlinear system, so we need an initial guess.
        #
        # We know the temperature `u` tends from `T0` to `T_ext` as t → ∞.
        # We also know the behavior is of an exponentially saturating type,
        # but we don't know the time constant (which, technically, isn't even a constant,
        # because `c = c(u)`). So let's be very rough, and use a linearly decreasing
        # temperature profile.
        u_.assign(project(Expression("T_ext + (1.0 - (x[0] / tmax)) * (T0 - T_ext)",
                                     degree=1,
                                     tmax=tmax, T0=T0, T_ext=T_ext),
                          W))

        n_system_iterations = 0
        maxit = 20
        tol = 1e-6
        converged = False
        while not converged:
            n_system_iterations += 1
            u_prev.assign(u_)

            # update nonlinear coefficient field
            c.assign(project(c_func(u_), W))

            # assemble and solve system
            A = assemble(a)
            b = assemble(L)
            [bc.apply(A) for bc in bcu]
            [bc.apply(b) for bc in bcu]
            solve(A, u_.vector(), b, 'mumps')
            # solve(A, u_.vector(), b, 'bicgstab', 'hypre_amg')  # more difficult to get to converge

            H1_diff = errornorm(u_, u_prev, "h1")
            if H1_diff < tol:
                converged = True
            if my_rank == 0:
                print(f"System iteration {n_system_iterations}, ‖u - u_prev‖_H1 = {H1_diff:0.6g}")
            if n_system_iterations > maxit:
                raise RuntimeError(f"System did not converge after {maxit} system iterations.")

        if my_rank == 0:
            print(f"System converged after {n_system_iterations} iterations.")

        # Extract the 1D array from the solution, using this approach:
        # https://fenicsproject.discourse.group/t/convert-fenics-solution-and-coordinates-to-numpy-2d-arrays/3104
        # x, = SpatialCoordinate(mesh)

        # # works in serial mode only
        # xx = W.tabulate_dof_coordinates()
        # assert xx.shape[1] == 1
        # xx = xx.flatten()
        # uu = u_.vector().get_local()

        # Let's use extrafeathers (works in MPI mode, too):
        cells, nodes = meshmagic.all_cells(W)
        dofs, nodes_array = meshmagic.nodes_to_array(nodes)
        xx = nodes_array.flatten()

        all_W_dofs = np.array(range(W.dim()), "intc")
        uu = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on W
        u_.vector().gather(uu, all_W_dofs)  # allgather

        perm = np.argsort(xx)  # sort the solution by increasing time coordinate
        return t0 + xx[perm], uu[perm]

    # Solve piecewise if necessary
    tmax_limit = 20.0  # [s]
    print("Solve for piece 0...")
    xx, uu = solve_piece(t0=0.0, tmax=min(tmax, tmax_limit), T0=T0)
    tol = 1e-8  # [K]
    last_piece = False
    if tmax > tmax_limit:
        xxs, uus = [xx], [uu]
        # Solve each piece
        j = 1
        while True:
            print(f"Solve for piece {j}...")
            piece_t0 = j * tmax_limit
            piece_tmax = (j + 1) * tmax_limit

            if piece_tmax >= tmax:
                piece_tmax = tmax
                last_piece = True

            xx, uu = solve_piece(t0=piece_t0, tmax=piece_tmax, T0=uu[-1])
            xxs.append(xx)
            uus.append(uu)

            if last_piece:
                break
            elif abs(uu[-1] - T_ext) <= tol:
                print("Tolerance reached, setting the end temperature as T_ext.")
                # Stop simulating once the temperature difference
                # to the external environment falls below `tol`
                xxs.append(np.array([tmax]))
                uus.append(np.array([T_ext]))
                break
            j += 1
        # Glue the pieces
        xx, *xxs = xxs
        uu, *uus = uus
        for xx_piece, uu_piece in zip(xxs, uus):
            if xx_piece[0] == xx[-1]:  # skip duplicated time value at the seams of pieces
                xx_piece = xx_piece[1:]
                uu_piece = uu_piece[1:]
            xx = np.concatenate((xx, xx_piece))
            uu = np.concatenate((uu, uu_piece))

    return xx, uu


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xx, uu = estimate(tmax=2000.0, nel=2000)
    if my_rank == 0:
        print(xx, uu)

        # visualize
        plt.plot(xx, uu)
        plt.grid(visible=True, which="both")
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"$T$ [K]")
        plt.title(f"Cooling of a sphere, R = {R:0.6g} m")
        plt.show()
