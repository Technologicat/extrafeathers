# -*- coding: utf-8; -*-
"""Axially moving solid, Eulerian view, small-displacement regime (on top of axial motion)."""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer, Popper

from fenics import (FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
                    DirichletBC,
                    Constant, Function,
                    project, Vector,
                    tr, Identity, sqrt, inner, dot,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end,
                    parameters)

# custom utilities for FEniCS
from extrafeathers import common
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import EulerianSolid
from extrafeathers.pdes.eulerian_solid import ε
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
Q = TensorFunctionSpace(mesh, 'P', 2)
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
# # TODO: Does not work with mass-lumped `u`, because in that algorithm we use only BCs for `v`.
# # TODO: If you want to try this case, set also the optional IC below.
# bcu_left = DirichletBC(V, Constant((-1e-3, 0)), boundary_parts, Boundaries.LEFT.value)
# bcu_right = DirichletBC(V, Constant((1e-3, 0)), boundary_parts, Boundaries.RIGHT.value)
# bcv_left = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.LEFT.value)  # ∂u/∂t
# bcv_right = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.RIGHT.value)  # ∂u/∂t
# bcu.append(bcu_left)
# bcu.append(bcu_right)
# bcv.append(bcv_left)
# bcv.append(bcv_right)

# # Optional: nonzero initial condition (IC) for displacement
# from fenics import Expression
# # u0 = project(Expression(("1e-3 * 2.0 * (x[0] - 0.5)", "0"), degree=1), V)  # [0, 1]
# u0 = project(Expression(("1e-3 * 2.0 * x[0]", "0"), degree=1), V)  # [-0.5, 0.5]
# solver.u_n.assign(u0)
# solver.u_.assign(u0)

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

# https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
parameters['krylov_solver']['nonzero_initial_guess'] = True
# parameters['krylov_solver']['monitor_convergence'] = True

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
all_Q_dofs = np.array(range(Q.dim()), "intc")
v_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
q_vec_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on Q

# TODO: We cannot export quads at full nodal resolution in FEniCS 2019,
# TODO: because the mesh editor fails with "cell is not orderable".
highres_export_V = (V.ufl_element().degree() > 1 and V.ufl_element().family() == "Lagrange")
if highres_export_V:
    if my_rank == 0:
        print("Preparing export of higher-degree u/v data as refined P1...")
    with timer() as tim:
        v_P1, my_V_dofs = meshmagic.prepare_linear_export(V)
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")
highres_export_Q = (Q.ufl_element().degree() > 1 and Q.ufl_element().family() == "Lagrange")
if highres_export_Q:
    if my_rank == 0:
        print("Preparing export of higher-degree σ data as refined P1...")
    with timer() as tim:
        q_P1, my_Q_dofs = meshmagic.prepare_linear_export(Q)
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
solver.stabilizers.SUPG = True  # stabilizer for advection-dominant problems

def dev(T):
    """Deviatoric part of rank-2 tensor `T`."""
    return T - (1 / 3) * tr(T) * Identity(T.geometric_dimension())

# W = FunctionSpace(mesh, "DP", 0)
# def elastic_strain_energy():
#     E = project(inner(solver.σ_, ε(solver.u_)),
#                 W,
#                 form_compiler_parameters={"quadrature_degree": 2})
#     return np.sum(E.vector()[:])
W = FunctionSpace(mesh, "R", 0)  # Function space of ℝ (single global DOF)
def elastic_strain_energy():
    "∫ σ : ε dΩ"
    return float(project(inner(solver.σ_, ε(solver.u_)), W))
def kinetic_energy():
    "∫ (1/2) ρ v² dΩ"
    # Note `solver._ρ`; we need the UFL `Constant` object here.
    return float(project((1 / 2) * solver._ρ * dot(solver.v_, solver.v_), W))

# Time-stepping
t = 0
msg = "Starting. Progress information will be available shortly..."
SUPG_str = "[SUPG] " if solver.stabilizers.SUPG else ""  # for messages
vis_step_walltime_local = 0
nsave_total = 1000  # how many timesteps to save from the whole simulation
nsavemod = int(nt / nsave_total)  # every how manyth timestep to save
vis_ratio = 0.01  # proportion of timesteps to visualize (plotting is slow)
nvismod = int(vis_ratio * nt)  # every how manyth timestep to visualize
est = ETAEstimator(nt, keep_last=nvismod)
if my_rank == 0:
    fig, ax = plt.subplots(2, 4, constrained_layout=True, figsize=(12, 6))
    plt.show()
    plt.draw()
    plotmagic.pause(0.001)
    colorbars = []
    print(f"Saving {nsave_total} timesteps in total -> save every {nsavemod} timestep{'s' if nsavemod > 1 else ''}.")
    print(f"Visualizing {100.0 * vis_ratio:0.3g}% of timesteps -> vis every {nvismod} timestep{'s' if nvismod > 1 else ''}.")
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    krylov_it1, krylov_it2, krylov_it3, (system_it, last_diff_H1) = solver.step()

    if n % nsavemod == 0 or n == nt - 1:
        begin("Saving")

        if highres_export_V:
            # Save the displacement visualization at full nodal resolution.
            solver.u_.vector().gather(v_vec_copy, all_V_dofs)  # allgather `u_` to `v_vec_copy`
            v_P1.vector()[:] = v_vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
            xdmffile_u.write(v_P1, t)

            # `v` lives on a copy of the same function space as `u`; recycle the temporary vector
            solver.v_.vector().gather(v_vec_copy, all_V_dofs)  # allgather `v_` to `v_vec_copy`
            v_P1.vector()[:] = v_vec_copy[my_V_dofs]  # LHS MPI-local; RHS global
            xdmffile_v.write(v_P1, t)
        else:  # save at P1 resolution
            xdmffile_u.write(solver.u_, t)
            xdmffile_v.write(solver.v_, t)

        if highres_export_Q:
            solver.σ_.vector().gather(q_vec_copy, all_Q_dofs)
            q_P1.vector()[:] = q_vec_copy[my_Q_dofs]
            xdmffile_σ.write(q_P1, t)
        else:  # save at P1 resolution
            xdmffile_σ.write(solver.σ_, t)

        # compute von Mises stress for visualization in ParaView
        # TODO: export von Mises stress at full nodal resolution, too
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
    visualize = n % nvismod == 0 or n == nt - 1
    if visualize:
        with timer() as tim:
            u_ = solver.u_
            v_ = solver.v_
            σ_ = solver.σ_

            def symmetric_vrange(p):
                ignored_minp, maxp = common.minmax(p, take_abs=True, mode="raw")
                return -maxp, maxp

            # remove old colorbars, since `plt.cla` doesn't
            if my_rank == 0:
                for cb in Popper(colorbars):
                    cb.remove()

            # u1
            if my_rank == 0:
                plt.sca(ax[0, 0])
                plt.cla()
            vmin, vmax = symmetric_vrange(u_.sub(0))
            theplot = plotmagic.mpiplot(u_.sub(0), prep=prep_V0, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$u_{1}$")
                plt.axis("equal")

            # u2
            if my_rank == 0:
                plt.sca(ax[0, 1])
                plt.cla()
            vmin, vmax = symmetric_vrange(u_.sub(1))
            theplot = plotmagic.mpiplot(u_.sub(1), prep=prep_V1, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$u_{2}$")
                plt.axis("equal")

            # v1
            if my_rank == 0:
                plt.sca(ax[1, 0])
                plt.cla()
            vmin, vmax = symmetric_vrange(v_.sub(0))
            theplot = plotmagic.mpiplot(v_.sub(0), prep=prep_V0, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$v_{1}$")
                plt.axis("equal")

            # v2
            if my_rank == 0:
                plt.sca(ax[1, 1])
                plt.cla()
            vmin, vmax = symmetric_vrange(v_.sub(1))
            theplot = plotmagic.mpiplot(v_.sub(1), prep=prep_V1, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$v_{2}$")
                plt.axis("equal")

            # σ11
            if my_rank == 0:
                plt.sca(ax[0, 2])
                plt.cla()
            vmin, vmax = symmetric_vrange(σ_.sub(0))
            theplot = plotmagic.mpiplot(σ_.sub(0), prep=prep_Q0, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$σ_{11}$")
                plt.axis("equal")

            # σ12
            if my_rank == 0:
                plt.sca(ax[0, 3])
                plt.cla()
            vmin, vmax = symmetric_vrange(σ_.sub(1))
            theplot = plotmagic.mpiplot(σ_.sub(1), prep=prep_Q1, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$σ_{12}$")
                plt.axis("equal")

            # σ21
            if my_rank == 0:
                plt.sca(ax[1, 2])
                plt.cla()
            vmin, vmax = symmetric_vrange(σ_.sub(2))
            theplot = plotmagic.mpiplot(σ_.sub(2), prep=prep_Q2, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$σ_{21}$")
                plt.axis("equal")

            # σ22
            if my_rank == 0:
                plt.sca(ax[1, 3])
                plt.cla()
            vmin, vmax = symmetric_vrange(σ_.sub(3))
            theplot = plotmagic.mpiplot(σ_.sub(3), prep=prep_Q3, show_mesh=True,
                                        cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if my_rank == 0:
                colorbars.append(plt.colorbar(theplot))
                plt.title(r"$σ_{22}$")
                plt.axis("equal")

            # figure title (progress message)
            if my_rank == 0:
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
    vis_step_walltime_local = nvismod * dt_avg

    E = elastic_strain_energy()
    K = kinetic_energy()
    if my_rank == 0:  # DEBUG
        print(f"Timestep {n + 1}/{nt}: Krylov {krylov_it1}, {krylov_it2}, {krylov_it3}; system {system_it}; ‖v - v_prev‖_H1 = {last_diff_H1}; E = ∫ σ:ε dΩ = {E:0.6g}; K = ∫ (1/2) ρ v² dΩ = {K:0.6g}; K + E = {K + E:0.6g}; wall time per timestep {dt_avg:0.6g}s; avg {1/dt_avg:0.3g} timesteps/sec (running avg, n = {len(est.que)})")

    # In MPI mode, one of the worker processes may have a larger slice of the domain
    # (or require more Krylov iterations to converge) than the root process.
    # So to get a reliable ETA, we must take the maximum across all processes.
    # But MPI communication is expensive, so only update this at vis steps.
    if visualize:
        times_global = MPI.comm_world.allgather((vis_step_walltime_local, est.estimate, est.formatted_eta))
        item_with_max_estimate = max(times_global, key=lambda item: item[1])
        max_eta = item_with_max_estimate[2]
        item_with_max_vis_step_walltime = max(times_global, key=lambda item: item[0])
        max_vis_step_walltime = item_with_max_vis_step_walltime[0]

    def roundsig(x, significant_digits):
        # https://www.adamsmith.haus/python/answers/how-to-round-a-number-to-significant-digits-in-python
        import math
        digits_in_int_part = int(math.floor(math.log10(abs(x)))) + 1
        decimal_digits = significant_digits - digits_in_int_part
        return round(x, decimal_digits)

    # msg for *next* timestep. Loop-and-a-half situation...
    msg = f"{SUPG_str}t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.6g}, {maxu:0.6g}]; {system_it} iterations; vis every {roundsig(max_vis_step_walltime, 2):g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

    # Loop-and-a-half situation, so draw one more time to update title.
    if visualize and my_rank == 0:
        # figure title (progress message)
        plt.suptitle(msg)
        # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
        plotmagic.pause(0.001)

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
