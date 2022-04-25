# -*- coding: utf-8; -*-
"""TODO: document this"""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (VectorFunctionSpace, TensorFunctionSpace, DirichletBC,
                    Expression, Constant, Function,
                    interpolate, project, Vector,
                    tr, Identity, sqrt, inner,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import common
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import EulerianSolid
from .config import (rho, lamda, mu, V0, dt, nt,
                     Boundaries,
                     mesh_filename,
                     vis_u_filename, sol_u_filename,
                     vis_v_filename, sol_v_filename,
                     vis_σ_filename, sol_σ_filename,
                     vis_vonMises_filename)

my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 1)
Q = TensorFunctionSpace(mesh, 'P', 1)
Vscalar = V.sub(0).collapse()
Qscalar = Q.sub(0).collapse()

if my_rank == 0:
    print(f"Number of DOFs: displacement {V.dim()}, velocity {V.dim()}, stress {Q.dim()}, total {2 * V.dim() + Q.dim()}")

bcu = []
bcv = []
bcσ = []
solver = EulerianSolid(V, Q, rho, lamda, mu, V0, bcu, bcv, bcσ, dt)  # Crank-Nicolson (default)
# solver = EulerianSolid(V, Q, rho, lamda, mu, V0, bcu, bcv, bcσ, dt, θ=1.0)  # backward Euler

# Define boundary conditions
#
# - on each boundary, set either `u` or `n·σ`
#   (the latter needs some trickery on boundaries not aligned with axes)
# - if you set `u`, set also `v` consistently

# Top and bottom edges: zero normal stress
bcσ_top1 = DirichletBC(Q.sub(1), Constant(0), boundary_parts, Boundaries.TOP.value)  # σ12 (symm.)
bcσ_top2 = DirichletBC(Q.sub(2), Constant(0), boundary_parts, Boundaries.TOP.value)  # σ21
bcσ_top3 = DirichletBC(Q.sub(3), Constant(0), boundary_parts, Boundaries.TOP.value)  # σ22
bcσ_bottom1 = DirichletBC(Q.sub(1), Constant(0), boundary_parts, Boundaries.BOTTOM.value)  # σ12
bcσ_bottom2 = DirichletBC(Q.sub(2), Constant(0), boundary_parts, Boundaries.BOTTOM.value)  # σ21
bcσ_bottom3 = DirichletBC(Q.sub(3), Constant(0), boundary_parts, Boundaries.BOTTOM.value)  # σ22
bcσ.append(bcσ_top1)
bcσ.append(bcσ_top2)
bcσ.append(bcσ_top3)
bcσ.append(bcσ_bottom1)
bcσ.append(bcσ_bottom2)
bcσ.append(bcσ_bottom3)

# # Left and right edges: fixed displacement at both ends
# bcu_left = DirichletBC(V, Constant((-1e-3, 0)), boundary_parts, Boundaries.LEFT.value)
# bcu_right = DirichletBC(V, Constant((1e-3, 0)), boundary_parts, Boundaries.RIGHT.value)
# bcv_left = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)  # ∂u/∂t
# bcv_right = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.RIGHT.value)  # ∂u/∂t
# bcu.append(bcu_left)
# bcu.append(bcu_right)
# bcv.append(bcv_left)
# bcv.append(bcv_right)

# Left and right edges: fixed left end, constant pull at right end (Kurki et al. 2016)
bcu_left = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)
bcv_left = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)  # ∂u/∂t
bcσ_right1 = DirichletBC(Q.sub(0), Constant(1), boundary_parts, Boundaries.RIGHT.value)  # σ11
bcσ_right2 = DirichletBC(Q.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value)  # σ12
bcσ_right3 = DirichletBC(Q.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value)  # σ21 (symm.)
bcu.append(bcu_left)
bcv.append(bcv_left)
bcσ.append(bcσ_right1)
bcσ.append(bcσ_right2)
bcσ.append(bcσ_right3)

# # Left and right edges: constant pull at both ends
# bcσ_left1 = DirichletBC(Q.sub(0), Constant(1), boundary_parts, Boundaries.LEFT.value)  # σ11
# bcσ_left2 = DirichletBC(Q.sub(1), Constant(0), boundary_parts, Boundaries.LEFT.value)  # σ12
# bcσ_left3 = DirichletBC(Q.sub(2), Constant(0), boundary_parts, Boundaries.LEFT.value)  # σ21 (symm.)
# bcσ_right1 = DirichletBC(Q.sub(0), Constant(1), boundary_parts, Boundaries.RIGHT.value)  # σ11
# bcσ_right2 = DirichletBC(Q.sub(1), Constant(0), boundary_parts, Boundaries.RIGHT.value)  # σ12
# bcσ_right3 = DirichletBC(Q.sub(2), Constant(0), boundary_parts, Boundaries.RIGHT.value)  # σ21 (symm.)
# bcσ.append(bcσ_left1)
# bcσ.append(bcσ_left2)
# bcσ.append(bcσ_left3)
# bcσ.append(bcσ_right1)
# bcσ.append(bcσ_right2)
# bcσ.append(bcσ_right3)

# # Optional: nonzero initial condition for displacement
# # u0 = project(Expression(("1e-3 * 2.0 * (x[0] - 0.5)", "0"), degree=1), V)  # [0, 1]
# u0 = project(Expression(("1e-3 * 2.0 * x[0]", "0"), degree=1), V)  # [-0.5, 0.5]
# solver.u_n.assign(u0)

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_v = XDMFFile(MPI.comm_world, vis_v_filename)
xdmffile_v.parameters["flush_output"] = True
xdmffile_v.parameters["rewrite_function_mesh"] = False

xdmffile_σ = XDMFFile(MPI.comm_world, vis_σ_filename)
xdmffile_σ.parameters["flush_output"] = True
xdmffile_σ.parameters["rewrite_function_mesh"] = False

# ParaView doesn't have a filter for this, so we compute it ourselves.
xdmffile_vonMises = XDMFFile(MPI.comm_world, vis_vonMises_filename)
xdmffile_vonMises.parameters["flush_output"] = True
xdmffile_vonMises.parameters["rewrite_function_mesh"] = False
vonMises = Function(Qscalar)

# Create time series (for use in other FEniCS solvers)
#
timeseries_u = TimeSeries(sol_u_filename)
timeseries_v = TimeSeries(sol_v_filename)
timeseries_σ = TimeSeries(sol_σ_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# HACK: Arrange things to allow exporting the velocity field at full nodal resolution.
all_V_dofs = np.array(range(V.dim()), "intc")
vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
if V.ufl_element().degree() > 1:
    if my_rank == 0:
        print("Preparing export of higher-degree data as refined P1...")
    with timer() as tim:
        func_P1, my_V_dofs = meshmagic.prepare_linear_export(V)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Analyze mesh and dofmap for plotting (static mesh, only need to do this once)
# `u` and `v` both live on `V`, so both can use the same preps.
prep_V0 = plotmagic.mpiplot_prepare(solver.u_.sub(0))
prep_V1 = plotmagic.mpiplot_prepare(solver.u_.sub(1))
prep_Q0 = plotmagic.mpiplot_prepare(solver.σ_.sub(0))
prep_Q1 = plotmagic.mpiplot_prepare(solver.σ_.sub(1))
prep_Q2 = plotmagic.mpiplot_prepare(solver.σ_.sub(2))
prep_Q3 = plotmagic.mpiplot_prepare(solver.σ_.sub(3))

# Enable stabilizers for the Galerkin formulation
solver.stabilizers.SUPG = False  # stabilizer for advection-dominant problems

def dev(T):
    """Deviatoric part of rank-2 tensor `T`."""
    return T - (1 / 3) * tr(T) * Identity(T.geometric_dimension())

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
SUPG_str = "[SUPG] " if solver.stabilizers.SUPG else ""  # for messages
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    krylov_it1, krylov_it2, krylov_it3, (v_it, e) = solver.step()
    if my_rank == 0:  # DEBUG
        print(f"Timestep {n + 1}/{nt}: Krylov {krylov_it1}, {krylov_it2}, {krylov_it3}; system {v_it}; ‖v - v_prev‖_H1 = {e}")

    begin("Saving")

    if V.ufl_element().degree() > 1:
        # Save the displacement visualization at full nodal resolution.
        solver.u_.vector().gather(vec_copy, all_V_dofs)  # allgather `u_` to `vec_copy`
        func_P1.vector()[:] = vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(func_P1, t)

        # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
        solver.v_.vector().gather(vec_copy, all_V_dofs)  # allgather `v_` to `vec_copy`
        func_P1.vector()[:] = vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_v.write(func_P1, t)
    else:  # save at P1 resolution
        xdmffile_u.write(solver.u_, t)
        xdmffile_v.write(solver.v_, t)
    xdmffile_σ.write(solver.σ_, t)

    # compute von Mises stress for visualization in ParaView
    s = dev(solver.σ_)
    vonMises_expr = sqrt(3 / 2 * inner(s, s))
    vonMises.assign(project(vonMises_expr, Qscalar))
    xdmffile_vonMises.write(vonMises, t)

    timeseries_u.store(solver.u_.vector(), t)  # the timeseries saves the original data
    timeseries_v.store(solver.v_.vector(), t)
    timeseries_σ.store(solver.σ_.vector(), t)
    end()

    # Accept the timestep, updating the "old" solution
    solver.commit()

    end()

    # Plot the components of u
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            if my_rank == 0:
                plt.figure(1)
                plt.clf()
                plt.subplot(2, 4, 1)
            u_ = solver.u_
            v_ = solver.v_
            σ_ = solver.σ_

            def get_symmetric_vrange(p):
                minp, maxp = common.minmax(p, take_abs=True, mode="raw")
                return maxp

            m = get_symmetric_vrange(u_.sub(0))
            theplot = plotmagic.mpiplot(u_.sub(0), prep=prep_V0, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$u_{1}$")
                plt.subplot(2, 4, 5)
            m = get_symmetric_vrange(u_.sub(1))
            theplot = plotmagic.mpiplot(u_.sub(1), prep=prep_V1, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$u_{2}$")
                plt.subplot(2, 4, 2)
            m = get_symmetric_vrange(v_.sub(0))
            theplot = plotmagic.mpiplot(v_.sub(0), prep=prep_V0, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$v_{1}$")
                plt.subplot(2, 4, 6)
            m = get_symmetric_vrange(v_.sub(1))
            theplot = plotmagic.mpiplot(v_.sub(1), prep=prep_V1, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$v_{2}$")
                plt.subplot(2, 4, 3)
            m = get_symmetric_vrange(σ_.sub(0))
            theplot = plotmagic.mpiplot(σ_.sub(0), prep=prep_Q0, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$σ_{11}$")
                plt.subplot(2, 4, 4)
            m = get_symmetric_vrange(σ_.sub(1))
            theplot = plotmagic.mpiplot(σ_.sub(1), prep=prep_Q1, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$σ_{12}$")
                plt.subplot(2, 4, 7)
            m = get_symmetric_vrange(σ_.sub(2))
            theplot = plotmagic.mpiplot(σ_.sub(2), prep=prep_Q2, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$σ_{21}$")
                plt.subplot(2, 4, 8)
            m = get_symmetric_vrange(σ_.sub(3))
            theplot = plotmagic.mpiplot(σ_.sub(3), prep=prep_Q3, show_mesh=True, cmap="RdBu_r", vmin=-m, vmax=+m)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.title(r"$σ_{22}$")
                plt.suptitle(msg)

            # info for msg (expensive; only update these once per vis step)

            # # On quad elements:
            # #  - `dolfin.interpolate` doesn't work (point/cell intersection only implemented for simplices),
            # #  - `dolfin.project` doesn't work for `dolfin.Expression`, either; same reason.
            # magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=V.ufl_element().degree(),
            #                        u0=u_.sub(0), u1=u_.sub(1))
            # magu = interpolate(magu_expr, Vscalar)
            # uvec = np.array(magu.vector())
            #
            # minu_local = uvec.min()
            # minu_global = MPI.comm_world.allgather(minu_local)
            # minu = min(minu_global)
            #
            # maxu_local = uvec.max()
            # maxu_global = MPI.comm_world.allgather(maxu_local)
            # maxu = max(maxu_global)

            # So let's do this manually. We can operate on the nodal values directly.
            minu, maxu = common.minmax(u_, mode="l2")

            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                plotmagic.pause(0.001)
        last_plot_walltime_local = tim.dt
        last_plot_walltime_global = MPI.comm_world.allgather(last_plot_walltime_local)
        last_plot_walltime = max(last_plot_walltime_global)

    # Update progress bar
    progress += 1

    # Do the ETA update as the very last thing at each timestep to include also
    # the plotting time in the ETA calculation.
    est.tick()
    # TODO: make dt, dt_avg part of the public interface in `unpythonic`
    dt_avg = sum(est.que) / len(est.que)
    vis_step_walltime_local = 50 * dt_avg

    # In MPI mode, one of the worker processes may have a larger slice of the domain
    # (or require more Krylov iterations to converge) than the root process.
    # So to get a reliable ETA, we must take the maximum across all processes.
    times_global = MPI.comm_world.allgather((vis_step_walltime_local, est.estimate, est.formatted_eta))
    item_with_max_estimate = max(times_global, key=lambda item: item[1])
    max_eta = item_with_max_estimate[2]
    item_with_max_vis_step_walltime = max(times_global, key=lambda item: item[0])
    max_vis_step_walltime = item_with_max_vis_step_walltime[0]

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{SUPG_str}t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.6g}, {maxu:0.6g}]; {v_it} iterations; vis every {max_vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
