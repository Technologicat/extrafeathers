# Changelog

**0.5.0** (last updated 19 May 2022):

*No user-visible changes yet.*


---


**0.4.0** (19 May 2022):

**Added**:

- **Quadrilateral elements** are now supported.
  - The support for quad elements in the legacy FEniCS itself is incomplete and buggy. It has been fixed in DOLFINx, but we do not support the next-gen FEniCS yet. So although `extrafeathers` already supports arbitrary quad meshes, this functionality is currently practically useful mostly for quad meshes on the unit square.
  - Like in FEniCS itself, mixed meshes (having both quads and triangles) are **not** supported.
- Add `prepare_linear_export`. See [`demo.coupled.main01_flow`](demo/coupled/main01_flow.py) and [`demo.boussinesq.main01_solve`](demo/boussinesq/main01_solve.py) for usage examples.
- Add `quad_to_tri` to convert a quad mesh to a triangle mesh in a crossed-diagonal format, by adding a node at each cell center and then replacing each quad by four triangles.
  - Note that emulating quad interpolation this way does not actually interpolate bilinearly; the interpolation remains linear on triangles. This produces minor visual artifacts when compared to actual bilinear interpolation. Still, the result looks ok-ish, and lets `mpiplot` (ab)use Matplotlib to plot FEM functions on quadrilaterals.
- Add `trimesh` to triangulate the unit square using equilateral triangles (halves of them at two opposite edges).
- Add `minmax` to extract the `min` and `max` of a FEM field on nodal elements. Modes available for raw, abs, l2 (euclidean length). When running in MPI mode, automatically gathers data from all processes. The data may live on an arbitrary subspace (e.g. a component of a vector field that itself lives on a `MixedElement`).
- Add `renumber_nodes_by_distance`. Can set the origin point.
- Add `collapse_node_numbering`. Like `dolfin.FunctionSpace.collapse`, but for the `extrafeathers` internal format (`cells` list and `nodes` dict, as produced by `all_cells`). This is sometimes needed; if curious, see the source code of the [`meshmagic`](extrafeathers/meshmagic.py) and [`plotmagic`](extrafeathers/plotmagic.py) modules for use cases.
- Add [interptest demo](demo/interptest.py), to show interpolation of a bilinear function on the unit square on different element types.
- Add [smoothing demo](demo/smoothingtest.py), to show the effects of P1->DP0->P1 (and Q1->DQ0->Q1) projection smoothing. This is a technique to eliminate a symmetric numerical checkerboard oscillation with only minor damage to the actual signal.
- Add [Eulerian solid mechanics demo](demo/euleriansolid/): axially moving Kelvin-Voigt sheet using an Eulerian description, with both dynamic and steady-state solvers.
  - This is actually a small open-source research code into the mechanics of axially moving materials, with applications in the analysis of industrial processes. Placing the code here facilitates co-evolving `extrafeathers` with features our research team currently actually needs.
  - Quadrilateral elements look nice for this.

**Changed**:

- Improved support for **discontinuous spaces** (DP1, DP2, DP3, DQ1, DQ2, and DQ3) in various library functions.
- `mpiplot` and `mpiplot_mesh` now support also piecewise constant spaces (`DP0` and `DQ0`).
- `mpiplot` now rejects input if the function space is not supported, instead of trying to project.
  - This is to ensure a faithful representation.
- Plotting preparation changed; now both `mpiplot` and `mpiplot_mesh` can take the `prep` argument.
  - Both `mpiplot_prepare` and `as_mpl_triangulation` generate a `prep`. These have slightly different options and different use cases.
  - `mpiplot` takes the output of `mpiplot_prepare`, and `mpiplot_mesh` that of `as_mpl_triangulation`.
- Rename `midpoint_refine` to `refine_for_export`, since that's the use, and it handles both degree-2 and degree-3 spaces.
- Rename `map_refined_P1` to `map_coincident`, and generalize it.
  - Now works at least with P1, P2, P3, Q1, Q2, Q3, DP0, DP1, DP2, DP3, DQ0, DQ1, DQ2, and DQ3 spaces.
  - Also, add a squared-distance tolerance option for detecting coincident nodes.


---


**0.3.0** (15 March 2022):

**Fixed**:

- `mpiplot` now works correctly for tensor/vector component functions (`mpiplot(u.sub(k))`)
  - The global DOF numbers are now correctly mapped to vertices of the Matplotlib triangulation.
- `mpiplot` now L2-projects (instead of interpolating) if the input is not P1, P2 or P3.
- `all_cells` no longer crashes if some MPI process has no cells.
- `as_mpl_triangulation`, `mpiplot_mesh` no longer crash if some MPI process has no triangles.
- **Fixes to solvers**:
  - [`pdes.navier_stokes`](extrafeathers/pdes/navier_stokes.py):
    - Zero out mean pressure only if no Dirichlet BCs on pressure (e.g. cavity flow test cases are like this).
      - Thus, now the pressure solution actually satisfies the Dirichlet BCs if any were given.
    - Fix crash during form compilation if the pressure space is dG0. Using that space doesn't work, because we need to take the gradient in the corrector steps of IPCS, but it shouldn't crash compilation.
  - `solver.step()` now returns the number of Krylov iterations taken. See [`pdes.navier_stokes`](extrafeathers/pdes/navier_stokes.py) and [`pdes.advection_diffusion`](extrafeathers/pdes/advection_diffusion.py) for details.
- **Fixes to demos**:
  - Fix Courant number computation.
  - Fix ETA computation: take maximum estimate across MPI processes.
  - Improve stabilizer status indicators.
  - Update statistics only at visualization steps, to run (up to 15%) faster.

**Changed**:

- `mpiplot` now has `show_mesh` and `show_partitioning` flags.
  - The mesh is drawn translucently on top of the function data, allowing to see at a glance whether the discretization looks fine or if more resolution is needed at some parts of the domain.
  - When *not* displaying the MPI partitioning, `mpiplot(u, show_mesh=True)` is more efficient than `mpiplot(u)` followed separately by `mpiplot_mesh(u.function_space().mesh())`, because it constructs the full nodal resolution Matplotlib triangulation just once.
- `all_cells`/`my_cells` now refines P3 meshes, too. Used by `mpiplot` and `midpoint_refine`, so these can now **refine P3 data** into full-nodal-resolution P1 for visualization and export. Each P3 triangle is split into nine P1 triangles, with an aesthetically pleasing fill.
- Demos involving **incompressible flow** now run also when **using a P1P1 discretization**, which is LBB-incompatible.
  - Based on a least-squares pressure smoothing technique to eliminate high spatial frequency numerical oscillations in the pressure field. Details in the docstring of [`pdes.navier_stokes.NavierStokes`](extrafeathers/pdes/navier_stokes.py), and in the demos [`demo.coupled.main01_flow`](demo/coupled/main01_flow.py) and [`demo.boussinesq.main01_solve`](demo/boussinesq/main01_solve.py).
  - P1P1 is not as accurate as P2P1 on the same mesh, but is much faster; may be useful for an initial investigation when there is a need to complete many ballpark simulations quickly.
  - The main point, however, is that now it is *possible* to choose a P1 space for the velocity field, should one desire to do so.
- Refactor [`pdes.advection_diffusion`](extrafeathers/pdes/advection_diffusion.py) into a generic constant-coefficient `AdvectionDiffusion`, and a separate, specific `HeatEquation`.
- Add a logo for the project.
- Tweak parameters of Boussinesq example.

**Added**:

- **New functions**:
  - `mpiplot_mesh` to plot the whole mesh in the root process.
    - The mesh is optionally color-coded by MPI partitioning.
    - The added subdivisions for P1 visualization of P2 and P3 fields can be optionally shown; by default, these are drawn in a more translucent color to distinguish them from element edges.
  - `patch_average` to compute the patch average of a P1/P2/P3 function. Works on scalars/vectors/tensors.
- **New demos**:
  - [`demo.dofnumbering`](demo/dofnumbering.py): visualize how FEniCS allocates its global DOFs (in 2D), both in serial and in parallel. Display also the MPI partitioning of the mesh.
  - [`demo.refelement`](demo/refelement.py): visualize the DOF numbering on a P2 or P3 element (in 2D), both locally (reference element) and globally (on a very small triangle mesh on the unit square).
    - MPI mode draws these diagrams individually for each process.
  - [`demo.patch_average`](demo/patch_average.py): demo of the new `patch_average` function.
  - [`demo.poisson_dg`](demo/poisson_dg.py): Poisson equation using symmetric interior penalty discontinuous Galerkin (SIPG) method.
    - Based on existing FEniCS demos and various internet sources. The main motivation of having this here is to collect the relevant information and links into one place. Each term of the variational problem is commented in detail, conceptually different but similar-looking terms are kept separate (e.g. Nitsche vs. dG stabilization), and the formulation accounts for a general Dirichlet BC. Comments also explain how to add Neumann and Robin BCs.
- **New other**:
  - [`countlines.py`](countlines.py) for project SLOC estimation. This is the same script as in `mcpyrate` and `unpythonic`.


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
