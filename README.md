# extrafeathers

Agility and ease-of-use batteries for the Python layer of the [FEniCS](https://fenicsproject.org/) finite element framework.

See examples in the [`demo/`](demo/) subfolder. Currently, we have MPI-enabled [Poisson](demo/poisson.py) and [Navier-Stokes](demo/navier_stokes.py) solvers, which are adapted, updated and extended from those in [the official FEniCS tutorial book](https://github.com/hplgit/fenics-tutorial).

We also demonstrate how to [import a Gmsh mesh](demo/import_gmsh.py) with subdomains (more than one physical surface in 2D), and then split it into individual meshes for different subproblems (this is useful e.g. for [FSI](https://en.wikipedia.org/wiki/Fluid%E2%80%93structure_interaction)). The physical boundary tags are automatically transferred from the full mesh.


## Features

 - **Mesh IO**
   - `import_gmsh`
     - Easily import a [Gmsh](https://gmsh.info/) mesh into FEniCS via [`meshio`](https://github.com/nschloe/meshio). Fire-and-forget convenience function, to cover the gap created by the deprecation of the old `dolfin-convert`.
     - Both 2D and 3D. Simplicial meshes (triangles, tetrahedra) only.
     - Outputs a single HDF5 file with three datasets: `/mesh`, `/domain_parts` (physical cells i.e. subdomains), and `/boundary_parts` (physical facets i.e. boundaries).
   - `read_hdf5_mesh`
     - Read in an imported mesh, and its physical cell and facet data.
   - `write_hdf5_mesh`
     - Write a mesh, and optionally its physical cell and facet data, in the same format as the output of `import_gmsh`.
 - **Mesh utilities**
   - `find_subdomain_boundaries`
     - Automatically tag facets on internal boundaries between two subdomains. Both 2D and 3D. This makes it easier to respect [DRY](https://en.wikipedia.org/wiki/Don't_repeat_yourself) when setting up a small problem for testing, as the internal boundaries only need to be defined in one place (in the actual geometry).
     - Tag also facets belonging to an outer boundary of the domain, via a callback function (that you provide) that gives the tag number for a given facet. This allows easily producing one `MeshFunction` with tags for all boundaries.
     - Here *subdomain* means a `SubMesh`. These may result either from internal mesh generation via the `mshr` component of FEniCS, or from imported meshes. See the `navier_stokes` and `import_gmsh` demos for examples of both.
   - `specialize_meshfunction`
     - Convert a `MeshFunction` on cells or facets of a full mesh into the corresponding `MeshFunction` on its `SubMesh`.
     - Both 2D and 3D. Cell and facet meshfunctions supported.
     - Useful e.g. when splitting a mesh with subdomains. This function allows converting the `domain_parts` and `boundary_parts` from the full mesh onto each submesh. This allows saving the submeshes, along with their subdomain and boundary tags, as individual standalone meshes in separate HDF5 mesh files. See the `import_gmsh` demo. This in turn is useful, because (as of FEniCS 2019) `SubMesh` is not supported when running in parallel.
   - `meshsize`
     - Compute the local mesh size (commonly denoted `h`), defined as the minimum edge length of each mesh entity. The result is returned as a `MeshFunction`.
     - Both 2D and 3D. Can compute both cell and facet meshfunctions.
     - Useful for stabilization methods in CFD, where `h` typically appears in the stabilization term.
 - **Plotting**
   - `mpiplot`
     - Plot the *whole* solution in the root process while running in parallel. As of v0.1.0, scalar field on a 2D triangle mesh only.
     - The full triangulation is automatically pieced together from all the MPI processes. For implementation simplicity, the visualization always uses linear triangle elements; other degrees are interpolated onto `P1`.
     - Often useful for debugging and visualizing simulation progress, especially for a lightweight MPI job that runs locally on a laptop (but still much faster with 4 cores rather than 1). Allows near-realtime visual feedback, and avoids the need to start [ParaView](https://www.paraview.org/) midway through the computation just to quickly check if the solver is still computing and if the results look reasonable.
   - `plot_facet_meshfunction`
     - Visualize whether the boundaries of a 2D mesh have been tagged as expected. Meant as a debug tool for use when generating and importing meshes. This functionality is oddly missing from `dolfin.plot`.


## Running the demos

With a terminal **in the top level directory of the project**, demos are run as Python modules. This will use the version of `extrafeathers` in the source tree (instead of an installed one, if any).

To run the **Poisson** demo,

```python
python -m demo.poisson  # serial
mpirun python -m demo.poisson  # parallel
```

To run the **Navier-Stokes** demo, with **uniform mesh generated via `mshr`**:

```python
python -m demo.navier_stokes  # serial mode = generate HDF5 mesh file
mpirun python -m demo.navier_stokes  # parallel mode = solve
```

To run the **Navier-Stokes** demo, with a **graded mesh imported from Gmsh**:

```python
python -m demo.import_gmsh  # generate HDF5 mesh file, overwriting the earlier one
mpirun python -m demo.navier_stokes
```

The Navier-Stokes demo supports solving only in parallel, because even a simple 2D [CFD](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) problem requires so much computing power that it makes no sense to run it serially on a garden-variety multicore laptop. Also, this way we can keep the script as simple as possible, and just abuse the MPI group size to decide what to do, instead of building a proper command-line interface using [`argparse`](https://docs.python.org/3/library/argparse.html).

### What's up with the Unicode variable names?

Looks more like math to use `ρ` instead of `rho`. Too bad Python doesn't accept `∇` or `∂` in variable names; with those, the PDEs would look [even better](https://github.com/gridap/Gridap.jl).

To type Unicode greek symbols, use an IME such as [latex-input](https://github.com/clarkgrubb/latex-input), or Emacs's `counsel-unicode-char` (from the [`counsel`](https://melpa.org/#/counsel) package; on its features, see this [blog post](https://oremacs.com/2015/04/09/counsel-completion/)).


## Dependencies

Beside the Python-based requirements in [`requirements.txt`](requirements.txt), this depends on `libhdf5` (backend for `h5py`) and `fenics`, which are not Python packages. You'll likely also want OpenMPI to run FEniCS on multiple cores (though `fenics` likely already pulls that in).

On Ubuntu-based systems,

```bash
sudo apt install libhdf5-dev libopenmpi-dev fenics
```

should install them. This was developed using `libhdf5-103`, `openmpi 4.0.3`, and `fenics 2019.2.0.5`.

If `h5py` fails to install, or crashes when trying to read/write HDF5 files, try recompiling it against the `libhdf5` headers you have; see the [build instructions](https://docs.h5py.org/en/stable/build.html#source-installation).

If you want to modify the `.geo` file and generate a new mesh for the Navier-Stokes demo, you'll need [Gmsh](https://gmsh.info/).

Additionally, [ParaView](https://www.paraview.org/) may be nice for visualizing the XDMF output files from FEniCS.


## Install & uninstall

### From source

Clone the repo from GitHub. Then, navigate to it in a terminal, and:

```bash
python -m setup install
```

possibly with `--user`, if your OS is a *nix, and you feel lucky enough to use the system Python. If not, activate your venv first; the `--user` flag is then not needed.

To uninstall:

```bash
pip uninstall extrafeathers
```

but first, make sure you're not in a folder that has an `extrafeathers` subfolder - `pip` will think it got a folder name instead of a package name, and become confused.


## License

All original code in this repository is licensed under [The Unlicense](LICENSE.md). Do whatever you want!

Any code fragments from forums are licensed by their respective authors under the terms the particular forum places on code contributions. In the case of StackOverflow, this means the fragments are used under the CC-BY-SA license. Attribution is given by providing the forum post URL and username in the source code comments.


## Thanks

Everyone who has posted solutions on the [old FEniCS Q&A forum](https://fenicsproject.org/qa/) (now archived) and the [discourse group](https://fenicsproject.discourse.group/); especially @Dokken.
