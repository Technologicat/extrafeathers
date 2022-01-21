# extrafeathers

Agility and ease-of-use batteries for the Python layer of the [FEniCS](https://fenicsproject.org/) finite element framework.

Usage examples can be found in the [`demo/`](demo/) subfolder. We currently have MPI-enabled examples for the [Poisson](demo/poisson.py) and [Navier-Stokes](demo/navier_stokes.py) problems, adapted, updated and extended from those in [the official FEniCS tutorial book](https://github.com/hplgit/fenics-tutorial).


## Features

 - **Plotting**
   - `mpiplot` plots the *whole* solution in the root process while running in MPI mode.
     - As of v0.1.0, scalar field on a 2D triangle mesh only.
     - The full triangulation is automatically pieced together from all the MPI processes. For implementation simplicity, the visualization always uses linear triangle elements; other degrees are interpolated onto `P1`.
     - Often useful for debugging and visualizing simulation progress, especially for a lightweight MPI job that runs locally on a laptop (but still much faster with 4 cores rather than 1). Allows near-realtime visual feedback, and avoids the need to start [ParaView](https://www.paraview.org/) midway through the computation just to quickly check if the solver is still computing and if the results look reasonable.
   - `plot_facet_meshfunction` can be used to visualize whether the boundaries of a 2D mesh have been tagged as expected. Meant as a debug tool for use when generating and importing meshes.
 - **Mesh generation**
   - `import_gmsh` imports a [Gmsh](https://gmsh.info/) mesh into FEniCS using [`meshio`](https://github.com/nschloe/meshio).
     - This is a fire-and-forget convenience function for the common use case, to cover the gap created by the deprecation of the old `dolfin-convert`. `meshio` is likely a better solution, but needs a simple interface for common simple tasks.
     - The output is a single HDF5 file with three datasets: `/mesh`, `/domain_parts` (physical cells), and `/boundary_parts` (physical facets). See `read_hdf5_mesh` below.
   - `find_subdomain_boundaries` automatically tags facets on internal boundaries between two subdomains. It should work for both 2D and 3D meshes.
     - If you provide a callback function, it can also tag facets belonging to an outer boundary of the domain. This allows easily producing one `dolfin.MeshFunction` that has tags for all boundaries.
     - Here *subdomain* means a `dolfin.SubMesh` generated using the FEniCS internal `mshr` mesh generation utility. See the `navier_stokes` demo for an example.
     - This makes it easier to respect [DRY](https://en.wikipedia.org/wiki/Don't_repeat_yourself) when setting up a small problem for testing, as the internal boundaries only need to be defined in one place (in the actual geometry).
   - The function `read_hdf5_mesh` reads in an imported mesh; for symmetry, we provide also `write_hdf5_mesh`.
   - `specialize_meshfunction` converts a `dolfin.MeshFunction` on cells or facets of a full mesh into the corresponding `dolfin.MeshFunction` on a `dolfin.SubMesh` of that mesh.
     - Useful e.g. for a mesh with subdomains for fluid and structure parts, to extract them and then save in separate HDF5 mesh files. See the `import_gmsh` demo.


## Running the demos

With a terminal **in the top level directory of the project**, demos can be run as Python modules, to use the version of `extrafeathers` in the source tree (instead of an installed one, if any).

To run the **Poisson** demo,

```python
python -m demo.poisson  # serial
mpirun python -m demo.poisson  # parallel
```

To run the **Navier-Stokes** demo, with **internally generated uniform mesh**:

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

Any original code in this repository is licensed under [The Unlicense](LICENSE.md). Do whatever you want!

Any code fragments from forums are licensed by their respective authors under the terms the particular forum places on code contributions. In the case of StackOverflow, this means the fragments are used under the CC-BY-SA license. Attribution is given by providing the forum post URL and username in the source code comments.


## Thanks

Everyone who has posted solutions on the [old FEniCS Q&A forum](https://fenicsproject.org/qa/) (now archived) and the [discourse group](https://fenicsproject.discourse.group/); especially @Dokken.
