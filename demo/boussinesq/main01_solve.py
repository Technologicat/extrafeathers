# -*- coding: utf-8; -*-
"""Boussinesq flow (natural convection). Two-way coupled problem."""

import numpy as np
import matplotlib.pyplot as plt

from unpythonic import ETAEstimator, timer

from fenics import (FunctionSpace, VectorFunctionSpace, DirichletBC,
                    Expression, Constant, Function,
                    interpolate, project, Vector,
                    XDMFFile, TimeSeries,
                    LogLevel, set_log_level,
                    Progress,
                    MPI,
                    begin, end)

# custom utilities for FEniCS
from extrafeathers import meshiowrapper
from extrafeathers import meshmagic
from extrafeathers import plotmagic

from extrafeathers.pdes import NavierStokes, HeatEquation
from .config import (rho, mu, c, k, alpha, T0, g, dt, nt,
                     Boundaries, L,
                     mesh_filename,
                     vis_T_filename, sol_T_filename,
                     vis_u_filename, sol_u_filename,
                     vis_p_filename, sol_p_filename)

my_rank = MPI.comm_world.rank

# Read mesh and boundary data from file
mesh, ignored_domain_parts, boundary_parts = meshiowrapper.read_hdf5_mesh(mesh_filename)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # velocity
Q = FunctionSpace(mesh, 'P', 1)  # pressure
W = FunctionSpace(mesh, 'P', 2)  # temperature

if my_rank == 0:
    print(f"Number of DOFs: velocity {V.dim()}, pressure {Q.dim()}, temperature {W.dim()}, total {V.dim() + Q.dim() + W.dim()}")

# Set up boundary conditions.
bcu_top = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.TOP.value)
bcu_walls = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.WALLS.value)
bcu_bottom = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.BOTTOM.value)
bcu_obstacle = DirichletBC(V, Constant((0, 0)), boundary_parts, Boundaries.OBSTACLE.value)
bcu = [bcu_top, bcu_walls, bcu_bottom, bcu_obstacle]

# # If you set the pressure at the top (outlet condition), then make sure *not*
# # to set a Dirichlet BC on `u` at the top.
# bcp_top = DirichletBC(Q, Constant(0.0), boundary_parts, Boundaries.TOP.value)
# bcp = [bcp_top]
bcp = []  # for cavity flows: Dirichlet BC on velocity everywhere, pure Neumann BCs on pressure.

# Also, if you set the pressure at the top, then don't set a temperature at the top here.
bcT_top = DirichletBC(W, Constant(0.0), boundary_parts, Boundaries.TOP.value)
bcT_walls = DirichletBC(W, Constant(0.0), boundary_parts, Boundaries.WALLS.value)
bcT_bottom = DirichletBC(W, Constant(0.0), boundary_parts, Boundaries.BOTTOM.value)
bcT_obstacle = DirichletBC(W, Constant(1.0), boundary_parts, Boundaries.OBSTACLE.value)
bcT = [bcT_top, bcT_walls, bcT_bottom, bcT_obstacle]

# Create XDMF files (for visualization in ParaView)
xdmffile_u = XDMFFile(MPI.comm_world, vis_u_filename)
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["rewrite_function_mesh"] = False

xdmffile_p = XDMFFile(MPI.comm_world, vis_p_filename)
xdmffile_p.parameters["flush_output"] = True
xdmffile_p.parameters["rewrite_function_mesh"] = False

xdmffile_T = XDMFFile(MPI.comm_world, vis_T_filename)
xdmffile_T.parameters["flush_output"] = True
xdmffile_T.parameters["rewrite_function_mesh"] = False

# Create time series (for use in other FEniCS solvers)
timeseries_u = TimeSeries(sol_u_filename)
timeseries_p = TimeSeries(sol_p_filename)
timeseries_T = TimeSeries(sol_T_filename)

# Create progress bar
progress = Progress('Time-stepping', nt)
# set_log_level(LogLevel.PROGRESS)  # use this to see the progress bar
set_log_level(LogLevel.WARNING)

plt.ion()

# Set up the problem
#
# TODO: Add support for resuming from the TimeSeries used to store `u_`, `p_`, and `T_`.
#
flowsolver = NavierStokes(V, Q, rho, mu, bcu, bcp, dt)
heatsolver = HeatEquation(W, rho, c, k, bcT, dt,
                          advection="divergence-free",
                          velocity_degree=V.ufl_element().degree())

# HACK: Arrange things to allow exporting P2/P3 fields at full nodal resolution.
if V.ufl_element().degree() > 1 or W.ufl_element().degree() > 1:
    if my_rank == 0:
        print("Preparing export of higher-degree data as refined P1...")
    with timer() as tim:
        if V.ufl_element().degree() > 1:
            if my_rank == 0:
                print("    Velocity...")
            u_P1, my_V_dofs = meshmagic.prepare_export_as_P1(V)
            all_V_dofs = np.array(range(V.dim()), "intc")
            u_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on V
        if W.ufl_element().degree() > 1:
            if my_rank == 0:
                print("    Temperature...")
            T_P1, my_W_dofs = meshmagic.prepare_export_as_P1(W)
            all_W_dofs = np.array(range(W.dim()), "intc")
            T_copy = Vector(MPI.comm_self)  # MPI-local, for receiving global DOF data on W
    if my_rank == 0:
        print(f"Preparation complete in {tim.dt:0.6g} seconds.")

# Analyze mesh and dofmap for plotting (static mesh, only need to do this once)
prep_Vcomp = plotmagic.mpiplot_prepare(Function(V.sub(0).collapse()))
prep_Q = plotmagic.mpiplot_prepare(flowsolver.p_)
prep_W = plotmagic.mpiplot_prepare(heatsolver.u_)

# Enable stabilizers for the Galerkin formulation
flowsolver.stabilizers.SUPG = True  # stabilizer for advection-dominant flows
flowsolver.stabilizers.LSIC = True  # additional stabilizer for high Re
heatsolver.stabilizers.SUPG = True  # stabilizer for advection-dominant problems

# Boussinesq buoyancy term (as an UFL expression).
#
# This goes into the flow solver as the specific body force.
α = Constant(alpha)
T = heatsolver.u_
ρ_over_ρ0 = 1 - α * (T - T0)  # nondimensional
specific_buoyancy = ρ_over_ρ0 * Constant((0, -g))  # N / kg = m / s²

# Time-stepping
t = 0
est = ETAEstimator(nt)
msg = "Starting. Progress information will be available shortly..."
flow_SUPG_str = "S" if flowsolver.stabilizers.SUPG else ""
flow_LSIC_str = "L" if flowsolver.stabilizers.LSIC else ""
flow_stabilizers_str = f"u[{flow_SUPG_str}{flow_LSIC_str}] " if any(flowsolver.stabilizers._as_dict().values()) else ""  # for messages
heat_stabilizers_str = "T[S] " if heatsolver.stabilizers.SUPG else ""  # for messages
vis_step_walltime_local = 0
for n in range(nt):
    begin(msg)

    # Update current time
    t += dt

    # Solve one timestep
    begin("Flow solve")
    flowsolver.f.assign(project(specific_buoyancy, V))
    flowsolver.step()
    flowsolver.commit()
    end()
    begin("Heat solve")
    heatsolver.a.assign(flowsolver.u_)
    heatsolver.step()
    heatsolver.commit()
    end()

    begin("Saving")

    if V.ufl_element().degree() > 1:
        # HACK: What we want to do:
        #   u_P1.assign(interpolate(flowsolver.u_, V_P1))
        # How we do it in MPI mode (see demo/coupled/main01_flow.py for full explanation):
        flowsolver.u_.vector().gather(u_copy, all_V_dofs)  # allgather into `u_copy`
        u_P1.vector()[:] = u_copy[my_V_dofs]  # LHS MPI-local; RHS global
        xdmffile_u.write(u_P1, t)
    else:
        xdmffile_u.write(flowsolver.u_, t)
    xdmffile_p.write(flowsolver.p_, t)
    timeseries_u.store(flowsolver.u_.vector(), t)  # the timeseries saves the original data
    timeseries_p.store(flowsolver.p_.vector(), t)

    if W.ufl_element().degree() > 1:
        heatsolver.u_.vector().gather(T_copy, all_W_dofs)  # allgather into `T_copy`
        T_P1.vector()[:] = T_copy[my_W_dofs]  # LHS MPI-local; RHS global
        xdmffile_T.write(T_P1, t)
    else:
        xdmffile_T.write(heatsolver.u_, t)
    timeseries_T.store(heatsolver.u_.vector(), t)

    end()

    end()

    # Plot p, the magnitude of u, and T
    if n % 50 == 0 or n == nt - 1:
        with timer() as tim:
            # Compute dynamic pressure min/max to center color scale on zero.
            pvec = np.array(flowsolver.p_.vector())

            minp_local = pvec.min()
            minp_global = MPI.comm_world.allgather(minp_local)
            minp = min(minp_global)

            maxp_local = pvec.max()
            maxp_global = MPI.comm_world.allgather(maxp_local)
            maxp = max(maxp_global)

            absmaxp = max(abs(minp), abs(maxp))

            if my_rank == 0:
                plt.figure(1)
                plt.clf()
                plt.subplot(1, 3, 1)
            theplot = plotmagic.mpiplot(flowsolver.p_, prep=prep_Q, cmap="RdBu_r", vmin=-absmaxp, vmax=+absmaxp)
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$p$")
                plt.subplot(1, 3, 2)
                plt.title(msg)
            magu_expr = Expression("pow(pow(u0, 2) + pow(u1, 2), 0.5)", degree=V.ufl_element().degree(),
                                   u0=flowsolver.u_.sub(0), u1=flowsolver.u_.sub(1))
            magu = interpolate(magu_expr, V.sub(0).collapse())
            theplot = plotmagic.mpiplot(magu, prep=prep_Vcomp, cmap="viridis")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$|u|$")
                plt.subplot(1, 3, 3)
            theplot = plotmagic.mpiplot(heatsolver.u_, prep=prep_W, cmap="coolwarm")
            if my_rank == 0:
                plt.axis("equal")
                plt.colorbar(theplot)
                plt.ylabel(r"$T$")

            # info for msg (expensive; only update these once per vis step)
            uvec = np.array(magu.vector())
            Tvec = np.array(heatsolver.u_.vector())

            minu_local = uvec.min()
            minu_global = MPI.comm_world.allgather(minu_local)
            minu = min(minu_global)

            maxu_local = uvec.max()
            maxu_global = MPI.comm_world.allgather(maxu_local)
            maxu = max(maxu_global)

            minT_local = Tvec.min()
            minT_global = MPI.comm_world.allgather(minT_local)
            minT = min(minT_global)

            maxT_local = Tvec.max()
            maxT_global = MPI.comm_world.allgather(maxT_local)
            maxT = max(maxT_global)

            # Courant number of *heat* solver
            #
            # For the advection term (model equation  ∂u/∂t + (a·∇)u = 0),
            # the Courant number is defined as:
            #    Co = |a| Δt / he
            # For the diffusion term (model equation  ∂u/∂t - ν ∇²u = 0),
            # it is defined as:
            #    Co = ν Δt / he²
            #
            # The idea behind the CFL condition is that the numerical scheme must
            # be able to access the information needed to determine the solution:
            # the numerical domain of dependence of a degree of freedom must contain
            # the corresponding analytical domain of dependence. Thus we should take
            # the maximum of these two Courant numbers.
            #
            # Note, however, that when using implicit time integration, each degree of freedom
            # actually has access to the solution in all of Ω, because the new value is solved
            # from a linear equation system that is spatially global.
            #
            # Indicentally, a similar consideration is also why it is sometimes said that the
            # concept of Courant number makes no sense for the *Navier-Stokes* equations of
            # incompressible flow. There is the ∇p term, which must be adjusted *globally*
            # at each timestep to satisfy ∇·u = 0. So the analytical domain of dependence
            # of each point of the solution regardless of the value of Δt is all of Ω.
            #
            # When using implicit methods, from a *stability* viewpoint it thus does not matter
            # if the Courant number is large. However, the Courant number still acts as an
            # indicator of *accuracy*. It is heuristically clear why: when time integration
            # is performed essentially by a finite difference approximation of the time
            # derivative, the method has no access to what should happen "between timesteps".
            # So especially in an advection problem, the numerical solution will drift away
            # from the exact solution if the timestep is too large, because the velocity
            # field is only sampled at the timesteps, and assumed to stay constant during
            # each timestep. Thus the streamlines of the numerical approximation of the
            # advection process will be piecewise linear. Obviously, if there are important
            # details in the velocity field an advected material parcel would hit between
            # timesteps (in the exact solution), these will be missed by the approximation.
            #
            # See e.g.
            #   https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
            #   https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method#Example:_2D_diffusion
            #
            # Note that in general, these expressions live on different function spaces,
            # because velocity and temperature may use different degree elements.
            Co_adv = project(magu_expr * Constant(dt) / flowsolver.he, V.sub(0).collapse())
            Co_dif = project(heatsolver.ν * Constant(dt) / heatsolver.he**2, W)
            maxCo_adv_local = np.array(Co_adv.vector()).max()
            maxCo_dif_local = np.array(Co_dif.vector()).max()
            maxCo_local = max(maxCo_adv_local, maxCo_dif_local)
            maxCo_global = MPI.comm_world.allgather(maxCo_local)
            maxCo = max(maxCo_global)

            # We don't have a freestream in this example, so let's use the maximum velocity.
            Re = flowsolver.reynolds(maxu, L)
            Pe = heatsolver.peclet(maxu, L)

            if my_rank == 0:
                plt.draw()
                if n == 0:
                    plt.show()
                # https://stackoverflow.com/questions/35215335/matplotlibs-ion-and-draw-not-working
                plotmagic.pause(0.2)
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
    msg = f"{flow_stabilizers_str}{heat_stabilizers_str}Re = {Re:0.2g}; Pe = {Pe:0.2g}; Co = {maxCo:0.2g}; t = {t + dt:0.6g}; Δt = {dt:0.6g}; {n + 2} / {nt} ({100 * (n + 2) / nt:0.1f}%); |u| ∈ [{minu:0.2g}, {maxu:0.2g}]; T ∈ [{minT:0.2g}, {maxT:0.2g}]; vis every {max_vis_step_walltime:0.2g} s (plot {last_plot_walltime:0.2g} s); {max_eta}"

# Hold plot
if my_rank == 0:
    plt.ioff()
    plt.show()
