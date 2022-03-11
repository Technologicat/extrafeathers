# Changelog

**0.3.0** (in progress, last updated 11 March 2022):

**Fixed**:

- `mpiplot` now works correctly for tensor/vector component functions (`mpiplot(u.sub(k))`)
- `mpiplot` now L2-projects (instead of interpolating) if the input is not P1, P2 or P3.
- **Fixes to solvers**:
  - `pdes.navier_stokes`: zero out mean pressure only if no Dirichlet BCs on pressure (e.g. cavity flow test cases are like this).
  - `solver.step()` now returns the number of Krylov iterations taken. See `pdes.navier_stokes` and `pdes.advection_diffusion` for details.
- **Fixes to demos**:
  - Fix Courant number computation.
  - Fix ETA computation: take maximum estimate across MPI processes.
  - Improve stabilizer status indicators.
  - Update statistics only at visualization steps, to run (up to 15%) faster.

**Changed**:

- `mpiplot` now has `show_mesh` and `show_partitioning` flags.
  - The mesh is drawn translucently on top of the function data, allowing to see at a glance whether the discretization looks fine or if more resolution is needed at some parts of the domain.
  - When *not* displaying the MPI partitioning, `mpiplot(u, show_mesh=True)` is more efficient than `mpiplot(u)` followed separately by `mpiplot_mesh(u.function_space().mesh())`, because it constructs the full nodal resolution Matplotlib triangulation just once.
- `all_cells`/`my_cells` now subdivides P3 meshes, too. Used by `mpiplot` and `midpoint_refine`, so these can now refine P3 data into full-nodal-resolution P1 for visualization and export.
- Demos involving incompressible flow now run also when using a P1P1 discretization, which is LBB-incompatible.
  - Based on a least-squares pressure smoothing technique to eliminate high spatial frequency numerical oscillations in the pressure field. Details in the docstring of `pdes.navier_stokes.NavierStokes`, and in the demos `demo.coupled.main01_flow` and `demo.boussinesq.main01_solve`.
  - P1P1 is not as accurate as P2P1 on the same mesh, but is much faster; may be useful for an initial investigation when there is a need to complete many ballpark simulations quickly.
  - The main point, however, is that now it is *possible* to choose a P1 space for the velocity field, should one desire to do so.
- Add a logo for the project.
- Tweak parameters of Boussinesq example.

**Added**:

- **New functions**:
  - `mpiplot_mesh` to plot the whole mesh in the root process.
    - The mesh is optionally color-coded by MPI partitioning.
    - The added subdivisions for P1 visualization of P2 and P3 fields can be optionally shown; by default, these are drawn in a more translucent color to distinguish them from element edges.
  - `patch_average` to compute the patch average of a P1/P2/P3 function. Works on scalars/vectors/tensors.
- **New demos**:
  - `demo.dofnumbering`: visualize how FEniCS allocates its global DOFs, both in serial and in parallel. Display also the MPI partitioning of the mesh.
  - `demo.refelement`: visualize the DOF numbering on a P2 or P3 element, both locally (reference element) and globally (on a very small triangle mesh on the unit square).
    - MPI mode draws these diagrams individually for each process.
  - `demo.patch_average`: demo of the new `patch_average` function.
  - `demo.poisson_dg`: Poisson equation using symmetric interior penalty discontinuous Galerkin (SIPG) method.
    - Based on existing FEniCS demos and various internet sources. The main motivation of having this here is to collect the relevant information and links into one place. Each term of the variational problem is commented in detail, conceptually different but similar-looking terms are kept separate (e.g. Nitsche vs. dG stabilization), and the formulation accounts for a general Dirichlet BC. Comments also explain how to add Neumann and Robin BCs.
- **New other**:
  - `countlines.py` for project SLOC estimation. This is the same script as in `mcpyrate` and `unpythonic`.

---

**0.2.0** (25 February 2022):

- Public API reorganized.
- Solvers now included as subpackage `extrafeathers.pdes`.
  - Not loaded automatically; if you need it, import explicitly.
- Stabilize Navier-Stokes and advection-diffusion solvers.
  - Skew-symmetric advection term for divergence-free velocity field for both.
  - SUPG (Streamline Upwinding Petrov-Galerkin) for both.
  - LSIC (Least squares incompressibility) for Navier-Stokes.
  - PSPG (Pressure stabilizing Petrov-Galerkin) for Navier-Stokes.
- Add Boussinesq flow (natural convection) demo, demonstrating a two-way coupled problem.
- License is now 2-clause BSD.


---

**0.1.0** (19 January 2022):

Initial version.
