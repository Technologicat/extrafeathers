# Changelog

**0.3.0** (in progress, last updated 25 February 2022):

- Add a logo for the project.
- Tweak parameters of Boussinesq example.


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
