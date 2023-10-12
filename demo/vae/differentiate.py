"""Numerically approximate the jacobian and hessian of a pixel image.

This module uses the weighted least squares meshfree (WLSQM) method, fitting a local quadratic surrogate model.
The derivatives of the surrogate are then approximations of the derivatives of the original tensor.

The method is also known as MLS, "moving least squares". The "moving" in the name is an anachronism; a classical
serial CPU implementation would solve the local systems one by one, thus explicitly "moving" the neighborhood.

This method produces approximations for both the jacobian and the hessian in one go, by solving local 5×5 equation systems.

We also provide a version that approximates the function value, the jacobian, and the hessian, by solving local 6×6 equation systems.

This is basically a technique demo; this can be GPU-accelerated with TF, so we can use it to evaluate the
spatial derivatives in the PDE residual in a physically informed loss for up to 2nd order PDEs.

A potential issue is that the VAE output might contain some noise, so to be safe, we need a method that can handle noisy input.

We need only the very basics here. A complete Cython implementation of WLSQM, and documentation:
    https://github.com/Technologicat/python-wlsqm
"""

__all__ = ["prepare", "solve",  # The hifiest algorithm, accurate, fast, uses a lot of VRAM. Recommended if you have the VRAM.
           "solve_lu", "solve_lu_custom",  # alternative solve step
           "multi_to_linear", "linear_to_multi",  # Conversion between meshgrid multi-index and linear index.
           "hifier_differentiate",  # Second best algorithm, not as accurate near corners, slower, uses less VRAM. Recommended for low-VRAM setups.
           "hifi_differentiate",   # Third best algorithm, a variant of the second.
           "differentiate",  # kernel for `hifi*`, can also be used separately for interior-only (fast), or with low-quality extrapolation at edges
           "fit_quadratic",  # kernel for `hifi*`; like `differentiate`, but compute function values too
           "fit_linear",  # kernel for `hifi*`; like `fit_quadratic`, but perform a locally linear fit instead
           "fit_constant",  # kernel for `hifi*`; like `fit_quadratic`, but perform a locally constant fit instead
           "coeffs_diffonly",  # for interpreting output of `differentiate`
           "coeffs_full"]  # for interpreting output of all other algorithms

import math
import typing

from unpythonic import timer

import numpy as np

import tensorflow as tf

from .padding import pad_quadratic_2d, pad_linear_2d
from .xgesv import solve_kernel as lusolve

coeffs_diffonly = {"dx": 0, "dy": 1, "dx2": 2, "dxdy": 3, "dy2": 4}  # for interpreting output of `differentiate`
coeffs_full = {"f": 0, "dx": 1, "dy": 2, "dx2": 3, "dxdy": 4, "dy2": 5}  # for interpreting output of all other algorithms


# TODO: move `sizeof_tensor` to a utility module
dtype_to_bytes = {tf.float16: 2,  # TODO: better way?
                  tf.float32: 4,
                  tf.float64: 8,
                  tf.int8: 1,
                  tf.int16: 2,
                  tf.int32: 4,
                  tf.int64: 8}
def sizeof_tensor(x: tf.Tensor, *, to_human: bool = True) -> int:
    """[debug utility] Get the size of `tf.Tensor` or `tf.RaggedTensor`, in bytes.

    This is the memory required for storing the actual data values.
    If you need to measure the Python object overhead, use `sys.getsizeof(x)`.

    `to_human`: If `True`, return results as a string in 10-based SI units
                (bytes, kB, MB, GB, TB).
                If `False`, return the raw int value in bytes.
    """
    if isinstance(x, tf.RaggedTensor):
        # https://www.tensorflow.org/api_docs/python/tf/experimental/DynamicRaggedShape
        storage_shape = tf.shape(x).inner_shape
    else:
        storage_shape = tf.shape(x)
    nel = int(tf.reduce_prod(storage_shape))
    size = nel * dtype_to_bytes[x.dtype]
    if not to_human:
        return size
    if size // int(1e12) > 0:  # future-proof mildly
        return f"{size / 1e12:0.3g} TB"
    if size // int(1e9) > 0:
        return f"{size / 1e9:0.3g} GB"
    if size // int(1e6) > 0:
        return f"{size / 1e6:0.3g} MB"
    if size // int(1e3) > 0:
        return f"{size / 1e3:0.3g} kB"
    return f"{size} bytes"


def measure_graph_size(f, *args):
    """[debug utility] Print how many nodes a given `@tf.function`, with given arguments, has in its graph.

    Automatically trace the graph first, if not done yet.

    Taken from:
        https://www.tensorflow.org/guide/function#looping_over_python_data
    """
    g = f.get_concrete_function(*args).graph
    print(f"{f.__name__}({', '.join(map(str, args))}) contains {len(g.as_graph_def().node)} nodes in its graph")


# --------------------------------------------------------------------------------
# Advanced API. Fastest GPU implementation. Best results. Needs most VRAM.
#
# Recommended if you have the VRAM.
# You most likely want just `prepare` and `solve`.

def multi_to_linear(iyix: tf.Tensor, *, shape: tf.Tensor):
    """[DOF mapper] Convert meshgrid multi-index to linear index.

    We assume C storage order (column index changes fastest).

    `iyix`: rank-2 tensor of meshgrid multi-indices, `[[iy0, ix0], [iy1, ix1], ...]`.
            Index offsets are fine, too.
    `shape`: rank-1 tensor, containing [ny, nx].

    Returns: rank-1 tensor of linear indices, `[idx0, idx1, ...]`.
             If input was offsets, returns offsets.
    """
    nx = int(shape[1])
    idx = iyix[:, 0] * nx + iyix[:, 1]
    return idx

def linear_to_multi(idx: tf.Tensor, *, shape: tf.Tensor):
    """[DOF mapper] Convert linear index to meshgrid multi-index.

    We assume C storage order (column index changes fastest).

    Signed index offsets cannot be recovered uniquely (e.g. for 256×256, should +511 be
    (+1, +255), or (+2, -1)?), so this only works for indices, not index offsets.

    `idx`: rank-1 tensor of linear indices, `[idx0, idx1, ...]`.
    `shape`: rank-1 tensor, containing [ny, nx].

    Returns: rank-2 tensor of meshgrid multi-indices, `[[iy0, ix0], [iy1, ix1], ...]`.
    """
    nx = int(shape[1])
    iy, ix = tf.experimental.numpy.divmod(idx, nx)  # TF 2.13
    return tf.stack([iy, ix], axis=1)  # -> [[iy0, ix0], [iy1, ix1], ...]


def _assemble_a(c: tf.Tensor,
                point_to_stencil: tf.Tensor,
                dtype: tf.DType,
                format: str,
                low_vram: bool,  # unused
                low_vram_batch_size: int) -> tf.Tensor:  # unused
    """[internal helper] Assemble the system matrix.

    The formula is::

      A[n,i,j] = ∑k( c[n,k,i] * c[n,k,j] )

    For the parameters, see `prepare`.

    Returns a `tf.Tensor` of shape [npoints, 6, 6].
    """
    @tf.function
    def _assemble_aij(c: tf.Tensor,
                      i: tf.Tensor,
                      j: tf.Tensor) -> tf.Tensor:
        # We upcast after slicing to save VRAM.
        # We always use float32 to compute the elements of `A` for optimal accuracy, even if our storage `dtype` happens to be float16.
        ci = tf.cast(c[:, :, i], tf.float32, name="cast_ci_to_float32")  # -> [#n, #k], where #k is ragged (number of neighbors in stencil for pixel `n`)
        cj = tf.cast(c[:, :, j], tf.float32, name="cast_cj_to_float32")  # -> [#n, #k]
        Aij = tf.reduce_sum(ci * cj, axis=1, name="assemble_aij")  # [#n, #k] -> [#n]
        return Aij

    # On a *uniform* meshgrid, it is enough to do the assembly for each stencil once.
    # Here the `c` that comes in has shape [#stencils, #k, 6], where #k is ragged.
    #
    # Even for large neighborhood sizes `N`, the number of unique stencils is
    # a couple thousand at most; hence we don't need to save VRAM here.
    #
    # # Python loop version:
    # rows = []
    # for i in range(6):
    #     row = []
    #     for j in range(6):
    #         Aij = _assemble_aij(c, tf.constant(i, dtype=tf.int64), tf.constant(j, dtype=tf.int64))
    #         row.append(tf.cast(Aij, dtype, name="cast_to_dtype"))
    #     row = tf.stack(row, axis=1)  # -> [#stencils, #cols]
    #     rows.append(row)
    # A = tf.stack(rows, axis=1)  # [[#stencils, #cols], [#stencils, #cols], ...] -> [#stencils, #rows, #cols]

    # We can still go a bit faster (for large stencil sizes) by accelerating the loops.
    # For small stencil sizes, this costs some compile time, but not much.
    # TODO: On a 6 GB card, runs out of VRAM in `reduce_sum` at `N=30.5, p=2.0` (256×256); without compiling this part, runs fine at `N=40.5`, still moderately fast.
    # @tf.function
    def _assemble():
        rows = tf.TensorArray(dtype, size=6)
        for i in tf.range(6):
            row = tf.TensorArray(dtype, size=6)
            for j in tf.range(6):
                Aij = _assemble_aij(c, i, j)  # -> [#stencils]
                row = row.write(j, tf.cast(Aij, dtype, name="cast_to_dtype"))
            row = row.stack()  # -> [#cols, #stencils]
            rows = rows.write(i, row)
        rows = rows.stack()  # -> [#rows, #cols, #stencils]
        A = tf.transpose(rows, [2, 0, 1])  # -> [#stencils, #rows, #cols]
        return A
    A = _assemble()

    # # DEBUG: If the x and y scalings work, the range of values in `A` should be approximately [0, (2 * N + 1)²].
    # # The upper bound comes from the maximal number of points in the stencil, and is reached when gathering this many "ones" in the constant term.
    # absA = tf.abs(A)
    # print(f"A[n, i, j] ∈ [{tf.reduce_min(absA):0.6g}, {tf.reduce_max(absA):0.6g}]")

    # TODO: DEBUG: sanity check that each `A[n, :, :]` is symmetric.

    if format == "A":
        # Replicate so that each pixel gets its own copy.
        # `A` doesn't take much VRAM, and it's easy to send this format to the linear system solver.
        A = tf.gather(A, point_to_stencil, axis=0)  # [#n, #rows, #cols]
        return A
    # format == "LUp":
    # Compute the LU decompositions before replicating.
    LU, perm = tf.linalg.lu(tf.cast(A, tf.float32))
    LU = tf.cast(LU, dtype)
    # Replicate.
    LU = tf.gather(LU, point_to_stencil, axis=0)
    perm = tf.gather(perm, point_to_stencil, axis=0)
    return LU, perm

    # # For documentation: assembly algorithm for general topologies.
    # # Here the `c` that comes in has shape [#n, #k, 6], where #k is ragged.
    # if not low_vram:
    #     # `einsum` doesn't support `RaggedTensor`. What we want to do:
    #     # # A = tf.einsum("nki,nkj->nij", c, c)
    #     # So we do it manually.
    #     # When enough VRAM is available, this is the simplest way to do the assembly:
    #     rows = []
    #     for i in range(6):
    #         ci = tf.cast(c[:, :, i], tf.float32, name="cast_ci_to_float32")  # -> [#n, #k], where #k is ragged (number of neighbors in stencil for pixel `n`)
    #         row = []
    #         for j in range(6):
    #             # Always use float32 to compute the elements of `A` for optimal accuracy, even if our storage `dtype` happens to be float16.
    #             cj = tf.cast(c[:, :, j], tf.float32, name="cast_cj_to_float32")  # -> [#n, #k]
    #             Aij = tf.reduce_sum(ci * cj, axis=1, name="assemble_aij")  # [#n, #k] -> [#n]
    #             row.append(tf.cast(Aij, dtype, name="cast_to_dtype"))
    #         row = tf.stack(row, axis=1)  # -> [#n, #cols]
    #         rows.append(row)
    # else:
    #     # In low VRAM mode, we assemble `A` in batches, splitting over the meshgrid points.
    #     # This is another straw that breaks the camel's back at N = 17.5 (with 6 GB VRAM).
    #
    #     # It significantly increases performance to split just this part off into a `tf.function`.
    #     # Converting all of `_assemble_a` into a `tf.function` slows down instead (or runs out of VRAM).
    #     #
    #     # To avoid triggering retracing, we wrap the Python `int` scalars `i` and `j` into TF tensors.
    #     # This runs slightly faster, even though we must then construct `tf.constant` tensors to send in scalars.
    #     @tf.function
    #     def _assemble_aij_batched(c_split: tf.Tensor,
    #                               i: tf.Tensor,
    #                               j: tf.Tensor) -> tf.Tensor:
    #         # We upcast after slicing to save VRAM.
    #         ci = tf.cast(c_split[:, :, i], tf.float32, name="cast_ci_to_float32")  # -> [#split, #k], where #k is ragged (number of neighbors in stencil for pixel `n`)
    #         cj = tf.cast(c_split[:, :, j], tf.float32, name="cast_cj_to_float32")  # -> [#split, #k]
    #         Aij = tf.reduce_sum(ci * cj, axis=1, name="assemble_aij")  # [#split, #k] -> [#split]
    #         return Aij
    #
    #     npoints = int(tf.shape(c)[0])  # not inside a @tf.function, so this is ok.
    #     n_batches = math.ceil(npoints / low_vram_batch_size)
    #     batches = [[j * low_vram_batch_size, (j + 1) * low_vram_batch_size] for j in range(n_batches)]
    #     batches[-1][-1] = npoints  # Set the final `stop` value to include all remaining tensor elements in the final batch.
    #     rows_by_batch = []
    #     for start, stop in batches:
    #         c_split = c[start:stop, :, :]  # [#n, #k, #cols] -> [#split, #k, #cols], where #k is ragged
    #         rows = []
    #         for i in range(6):
    #             row = []
    #             for j in range(6):
    #                 Aij = _assemble_aij_batched(c_split,
    #                                             tf.constant(i, dtype=tf.int64),
    #                                             tf.constant(j, dtype=tf.int64))  # [#split]
    #                 row.append(tf.cast(Aij, dtype, name="cast_to_dtype"))
    #             row = tf.stack(row, axis=1)  # -> [#split, #cols]
    #             rows.append(row)
    #         rows_by_batch.append(rows)
    #     # Unsplit:
    #     #   [[batch0_row0, batch0_row1, ..., batch0_row5], [batch1_row0, batch1_row1, ..., batch1_row5], ...] -> [row0, row1, ..., row5]
    #     # `A` itself doesn't take much VRAM (less than 10 MB typically), so we don't have to worry about VRAM usage here.
    #     rows = []
    #     for rs in zip(*rows_by_batch):
    #         rows.append(tf.concat(rs, axis=0))  # [#split, #cols] -> [#n, #cols]
    # A = tf.stack(rows, axis=1)  # [[#n, #cols], [#n, #cols], ...] -> [#n, #rows, #cols]
    # return A

def prepare(N: float,
            X: typing.Union[np.array, tf.Tensor],
            Y: typing.Union[np.array, tf.Tensor],
            Z: typing.Union[np.array, tf.Tensor],
            *,
            p: typing.Union[float, str] = 2.0,
            dtype: tf.DType = tf.float32,
            format: str = "A",
            low_vram: bool = True,
            low_vram_batch_size: int = 8192,
            print_statistics: bool = False,
            indent: str = ""):
    """Prepare for differentiation on a meshgrid.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    This is the hifiest algorithm provided in this module. For what to do after `prepare`, see `solve`.

    This function precomputes the surrogate fitting coefficient tensor `c`, and the pixelwise `A` matrices.
    Some of the preparation is performed on the CPU, the most intensive computations on the GPU.

    NOTE: For large `N` and/or large input image resolution, this can take a lot of VRAM.

    `N`: Neighborhood size parameter (in grid spacings).

         The edges and corners of the input image are handled by clipping the stencil to the data region
         (to use as much data for each pixel as exists within distance `N` on each axis).

         Non-integer values can be useful with float values of `p` (see below). This affects the shape
         of the outer edge of the neighborhood. (E.g., with `p = 2.0`, integer `N` has a 1-pixel protrusion
         along the cardinal directions, although otherwise the stencil is a good approximation of a round shape.
         Using e.g. `N=11.5` instead of `N=11` can avoid this.)

    `X`, `Y`, `Z`: data in meshgrid format for x, y, and function value, respectively.
                   The shapes of `X`, `Y` and `Z` must match.

                   `Z` is only consulted for its shape and dtype.

                   The grid spacing must be uniform.

    `p`: The p for p-norm that is used for deciding whether a grid point in the local [-N, N]²
         box belongs to the stencil.

         Either a float >= 1.0, or the string "inf" (p-infinity norm, keep the whole box).

         Particularly useful is the default `p=2.0`, i.e. Euclidean distance, yielding round neighborhoods.
         This is the most isotropic possible, avoiding directional artifacts when the surrogate fitter is
         used for denoising noisy function data.

    `dtype`: The desired TensorFlow data type for the outputs `A`, `c`, and `scale`.
             The output `neighbors` always has dtype `int32`.

    `format`: One of "A" or "LUp":
        "A": Format expected by `solve`. Recommended.
             The returned `tensors` (see below) are `(A, c, scale, point_to_stencil, stencils, neighbors)`.

        "LUp": Format expected by `solve_lu` (which can run also at float16).
             The returned `tensors` (see below) are `(LU, perm, c, scale, point_to_stencil, stencils, neighbors)`.

        The returned tensors can be passed to `solve` or `solve_lu` (depending on `format`);
        these solvers run completely on the GPU.

    `low_vram`: If `True`, perform the following:

                  - Store the coefficient tensor `c` in half precision (float16).
                    This makes `A` slightly less precise, but not much, while it halves the VRAM
                    requirements for the largest tensor in the preparation process.

                  - Assemble `A` in batches (over meshgrid points). See the `low_vram_batch_size` parameter.

    `low_vram_batch_size`: If `low_vram=True`, this is the batch size for assembling the system matrix.
                           Decrease this to trade off speed for lower VRAM usage.

                           If `low_vram=False`, this parameter is ignored.

    `print_statistics`: If `True`, print on stdout how much memory (usually VRAM) the
                        various tensors used during the preparation process use.

                        This of course only succeeds if there is enough memory to
                        actually allocate those tensors. The idea is to help give
                        a rough intuition of the scaling behavior.

                        This option also prints wall time taken for various stages
                        of preparation.

    `indent`: If `print_statistics=True`, prefix string for messages, usually some number of spaces.
              If `print_statistics=True`, this parameter is ignored.

    As long as `N`, `X` and `Y` remain constant, and the dtype of `Z` remains the same,
    the same preparation can be reused.

    The complete return value is `(tensors, stencil)`, where:
      - `tensors` is as explained under the parameter `format`,
      - `stencil` is a rank-2 np-array `[[iy0, ix0], [iy1, ix1], ...]`, containing the multi-index offsets (int)
         of grid points belonging to the stencil in the interior part of the image. This is intended
         mainly for information (e.g. `len(stencil)` gives the number of neighbors used), and for plotting
         an example stencil onto a meshgrid plot. Choose a grid point `[i, j]`, and then::

             idxs = np.array([i, j]) + stencil
             plt.scatter(X[idxs[:, 0], idxs[:, 1]], Y[idxs[:, 0], idxs[:, 1]], c="k", marker="o")
    """
    if N < 1:
        raise ValueError(f"Must specify N ≥ 1, got {N}")
    # The image must be at least [2 * N + 1, 2 * N + 1] pixels, to have at least one pixel in the interior, so that our division into regions is applicable.
    for name, tensor in (("X", X), ("Y", Y), ("Z", Z)):
        shape = tf.shape(tensor)
        if len(shape) != 2:
            raise ValueError(f"Expected `{name}` to be a rank-2 tensor; got rank {len(shape)}")
        if not (int(shape[0]) >= 2 * N + 1 and int(shape[1]) >= 2 * N + 1):
            raise ValueError(f"Expected `{name}` to be at least of size [(2 * N + 1) (2 * N + 1)]; got N = {N} and {shape}")
    if not ((isinstance(p, float) and p >= 1.0) or (isinstance(p, str) and p == "inf")):
        raise ValueError(f"Expected `p` to be either a float >= 1 or the string 'inf'; got {p}")
    if format not in ("A", "LUp"):
        raise ValueError(f"Unknown format '{format}'; known: 'A', 'LUp'.")

    def intarray(x):
        return np.array(x, dtype=np.int32)

    # Allow fractional radius, but build box for the next integer size.
    radius = N
    N = math.ceil(N)

    if p == "inf":
        # We know that |iy| ≤ N, |ix| ≤ N in the whole box, so we could skip the check,
        # but it's a better contract if the function that is supposed to do checking actually checks.
        def belongs_to_neighborhood(iy, ix):  # infinity norm
            return max(abs(iy), abs(ix)) <= radius
    else:  # isinstance(p, float) and p >= 1.0:
        def belongs_to_neighborhood(iy, ix):  # p-norm, general case
            return (iy**p + ix**p)**(1 / p) <= radius

    # --------------------------------------------------------------------------------
    # Topology-specific stencil assembly

    shape = tf.shape(Z)
    npoints = tf.reduce_prod(shape)  # not inside a @tf.function, so this is ok.

    if print_statistics:
        print(f"{indent}Data tensor size: {shape.numpy()} ({npoints} pixels)")
        print(f"{indent}Batch size: {low_vram_batch_size} pixels per batch (⇒ {math.ceil(npoints / low_vram_batch_size)} batches)")
        if low_vram_batch_size > npoints:
            print(f"{indent}    Limiting batch size to data size {npoints}.")
    low_vram_batch_size = min(npoints, low_vram_batch_size)  # can't have a batch larger than there is data

    with timer() as tim:
        if print_statistics:
            print(f"{indent}Assemble stencils for N = {radius:0.3g}, p = {p:0.3g}...")
        all_multi_to_linear = tf.reshape(tf.range(npoints), shape)  # e.g. [[0, 1, 2], [3, 4, 5], ...]; C storage order assumed

        # Adapt the uniform [2 * N + 1, 2 * N + 1] stencil for each pixel, by clipping it into the data region.
        # Note that clipping to the data region is only needed near edges and corners.
        #
        # To save some VRAM, we store only unique stencils. Because we work on a meshgrid:
        #  1) We can store each unique stencil as a list of linear index *offsets*, taking advantage of the uniform grid topology,
        #  2) When using such as offset format, only relatively few unique stencils appear.
        #
        # To handle edges and corners optimally, using as much of the data as possible, each corner needs a different stencil
        # for each pixel (N² of them), and the remaining part of each row or column (N of them) near an edge needs a different stencil.
        # So in total, we have 1 + 4 * N + 4 * N² unique stencils (where the 1 is for the interior). Crucially, this is independent
        # of the input image resolution.
        #
        # We create a per-pixel indirection tensor, mapping pixel linear index to the index of the appropriate unique stencil.
        #
        # For large stencils (e.g. at `N = 8`, (2 * N + 1)² = 289 points), the indirection saves a lot of VRAM, mainly because
        # we only need to store one copy of the stencil for the interior part, which makes up most of the input image.
        #
        # A stencil consists of int32 entries, so *without* indirection, for example for `N = 8`, with naive per-pixel stencils,
        # the amount of VRAM we would need is 289 * 4 bytes * resolution², which is 303 MB at 512×512, and 1.21 GB at 1024×1024.
        # *With* indirection, we save the factor of 289, since we need only one int32 per pixel to identify which stencil to use.
        # Thus 4 bytes * resolution², which is 1.1 MB at 512×512, and 4.2 MB at 1024×1024.
        #
        # Of course, we need to store the actual unique stencils, too. But the upper bound for memory use for this is (stencil size) * (n_stencils) * 4 bytes
        # = (2 * N + 1)² * (1 + 4 * N + 4 * N²) * 4 bytes; actual memory use is less than this due to the clipping to data region. For `N = 8`, this is
        # 289 * 289 * 4 bytes = 83521 * 4 bytes = 334 kB, so the storage for the unique stencils needs very little memory.
        stencils = []  # list of lists; we will convert to a ragged tensor at the end
        point_to_stencil = np.zeros([npoints], dtype=int) - 1  # -1 = uninitialized, to catch bugs
        stencil_to_points = []
        def register_stencil(stencil: np.array, for_points: tf.Tensor) -> int:  # set up indirection
            stencils.append(stencil)
            stencil_id = len(stencils) - 1
            point_to_stencil[for_points] = stencil_id
            stencil_to_points.append(list(for_points.numpy()))  # stencil id -> list of points
            return stencil_id

        # # Turned out that for this use case, the best acceleration tool is NumPy. See below.
        # def _belongs(iy, ix):  # p-norm, general case
        #     x = tf.cast(iy, tf.float32)
        #     y = tf.cast(ix, tf.float32)
        #     return tf.less_equal((y**p + x**p)**(1 / p), radius)
        # @tf.function
        # def _build_stencil(ystart, ystop, xstart, xstop):
        #     # # Slower than Python.
        #     # out = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        #     # j = 0
        #     # for iy in tf.range(ystart, ystop):
        #     #     for ix in tf.range(xstart, xstop):
        #     #         if belongs(iy, ix):
        #     #             out = out.write(j, tf.stack([iy, ix]))  # multi-index offset
        #     #             j += 1
        #     # out = out.stack()
        #     # return out
        #     #
        #     # Still slower than Python.
        #     iy = tf.range(ystart, ystop)  # [ny]
        #     ix = tf.range(xstart, xstop)  # [nx]
        #     X, Y = tf.meshgrid(ix, iy)  # [ny, nx]
        #     x = tf.reshape(X, [-1])
        #     y = tf.reshape(Y, [-1])
        #     mask = tf.where(_belongs(y, x), 1, 0)
        #     yx = tf.stack([y, x], axis=1)
        #     return tf.boolean_mask(yx, mask)

        # Interior - one stencil for all pixels; this case handles almost all of the image.
        if print_statistics:
            print(f"{indent}    Interior (1 stencil)...")
        interior_multi_to_linear = all_multi_to_linear[N:-N, N:-N]  # take the interior part of the meshgrid
        interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior pixel (C storage order)
        # We build just this one stencil manually.
        # interior_stencil = _build_stencil(ystart=tf.constant(-N), ystop=tf.constant(N + 1), xstart=tf.constant(-N), xstop=tf.constant(N + 1))
        interior_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                              for ix in range(-N, N + 1)
                                              if belongs_to_neighborhood(iy, ix)])  # multi-index offsets
        orig_interior_stencil = interior_stencil  # for returning to caller
        if print_statistics:
            print(f"{indent}        {len(orig_interior_stencil)} points")
        interior_stencil = multi_to_linear(interior_stencil, shape=shape)  # corresponding linear index offsets
        register_stencil(interior_stencil, interior_idx)

        # Top edge - one stencil per row (N of them, so typically 8).
        if print_statistics:
            print(f"{indent}    Top edge ({N} stencils)...")
        for row in range(N):
            top_multi_to_linear = all_multi_to_linear[row, N:-N]
            top_idx = tf.reshape(top_multi_to_linear, [-1])
            # top_stencil = _build_stencil(ystart=-row, ystop=tf.constant(N + 1), xstart=tf.constant(-N), xstop=tf.constant(N + 1))
            # top_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
            #                                  for ix in range(-N, N + 1)
            #                                  if belongs_to_neighborhood(iy, ix)])
            # top_stencil = multi_to_linear(top_stencil, shape=shape)
            # The interior stencil already accounts for the desired shape (p-norm `p`). We can just quickly cut it to the data region using NumPy.
            top_stencil = interior_stencil[orig_interior_stencil[:, 0] >= -row]
            register_stencil(top_stencil, top_idx)  # each row near the top gets its own stencil

        # Bottom edge - one stencil per row.
        if print_statistics:
            print(f"{indent}    Bottom edge ({N} stencils)...")
        for row in range(-N, 0):
            bottom_multi_to_linear = all_multi_to_linear[row, N:-N]
            bottom_idx = tf.reshape(bottom_multi_to_linear, [-1])
            # bottom_stencil = _build_stencil(ystart=tf.constant(-N), ystop=-row, xstart=tf.constant(-N), xstop=tf.constant(N + 1))
            # bottom_stencil = intarray([[iy, ix] for iy in range(-N, -row)
            #                                     for ix in range(-N, N + 1)
            #                                     if belongs_to_neighborhood(iy, ix)])
            # bottom_stencil = multi_to_linear(bottom_stencil, shape=shape)
            bottom_stencil = interior_stencil[orig_interior_stencil[:, 0] < -row]
            register_stencil(bottom_stencil, bottom_idx)

        # Left edge - one stencil per column (N of them, so typically 8).
        if print_statistics:
            print(f"{indent}    Left edge ({N} stencils)...")
        for col in range(N):
            left_multi_to_linear = all_multi_to_linear[N:-N, col]
            left_idx = tf.reshape(left_multi_to_linear, [-1])
            # left_stencil = _build_stencil(ystart=tf.constant(-N), ystop=tf.constant(N + 1), xstart=-col, xstop=tf.constant(N + 1))
            # left_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
            #                                   for ix in range(-col, N + 1)
            #                                   if belongs_to_neighborhood(iy, ix)])
            # left_stencil = multi_to_linear(left_stencil, shape=shape)
            left_stencil = interior_stencil[orig_interior_stencil[:, 1] >= -col]
            register_stencil(left_stencil, left_idx)

        # Right edge - one stencil per column.
        if print_statistics:
            print(f"{indent}    Right edge ({N} stencils)...")
        for col in range(-N, 0):
            right_multi_to_linear = all_multi_to_linear[N:-N, col]
            right_idx = tf.reshape(right_multi_to_linear, [-1])
            # right_stencil = _build_stencil(ystart=tf.constant(-N), ystop=tf.constant(N + 1), xstart=tf.constant(-N), xstop=-col)
            # right_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
            #                                    for ix in range(-N, -col)
            #                                    if belongs_to_neighborhood(iy, ix)])
            # right_stencil = multi_to_linear(right_stencil, shape=shape)
            right_stencil = interior_stencil[orig_interior_stencil[:, 1] < -col]
            register_stencil(right_stencil, right_idx)

        # Upper left corner - one stencil per pixel (N² of them, so typically 64).
        if print_statistics:
            print(f"{indent}    Upper left corner ({N * N} stencils)...")
        for row in range(N):
            for col in range(N):
                this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])  # just one pixel, but for uniform data format, use a rank-1 tensor
                # this_stencil = _build_stencil(ystart=-row, ystop=tf.constant(N + 1), xstart=-col, xstop=tf.constant(N + 1))
                # this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                #                                   for ix in range(-col, N + 1)
                #                                   if belongs_to_neighborhood(iy, ix)])
                # this_stencil = multi_to_linear(this_stencil, shape=shape)
                this_stencil = interior_stencil[(orig_interior_stencil[:, 0] >= -row) * (orig_interior_stencil[:, 1] >= -col)]
                register_stencil(this_stencil, this_idx)

        # Upper right corner - one stencil per pixel.
        if print_statistics:
            print(f"{indent}    Upper right corner ({N * N} stencils)...")
        for row in range(N):
            for col in range(-N, 0):
                this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
                # this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                #                                   for ix in range(-N, -col)
                #                                   if belongs_to_neighborhood(iy, ix)])
                # this_stencil = multi_to_linear(this_stencil, shape=shape)
                this_stencil = interior_stencil[(orig_interior_stencil[:, 0] >= -row) * (orig_interior_stencil[:, 1] < -col)]
                register_stencil(this_stencil, this_idx)

        # Lower left corner - one stencil per pixel.
        if print_statistics:
            print(f"{indent}    Lower left corner ({N * N} stencils)...")
        for row in range(-N, 0):
            for col in range(N):
                this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
                # this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                #                                   for ix in range(-col, N + 1)
                #                                   if belongs_to_neighborhood(iy, ix)])
                # this_stencil = multi_to_linear(this_stencil, shape=shape)
                this_stencil = interior_stencil[(orig_interior_stencil[:, 0] < -row) * (orig_interior_stencil[:, 1] >= -col)]
                register_stencil(this_stencil, this_idx)

        # Lower right corner - one stencil per pixel.
        if print_statistics:
            print(f"{indent}    Lower right corner ({N * N} stencils)...")
        for row in range(-N, 0):
            for col in range(-N, 0):
                this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
                # this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                #                                   for ix in range(-N, -col)
                #                                   if belongs_to_neighborhood(iy, ix)])
                # this_stencil = multi_to_linear(this_stencil, shape=shape)
                this_stencil = interior_stencil[(orig_interior_stencil[:, 0] < -row) * (orig_interior_stencil[:, 1] < -col)]
                register_stencil(this_stencil, this_idx)

        assert len(stencils) == 1 + 4 * N + 4 * N**2  # interior, edges, corners
        assert not (point_to_stencil == -1).any()  # every pixel of the input image should now have a stencil associated with it
    if print_statistics:
        print(f"{indent}    Assembled {len(stencils)} stencils in total.")
        print(f"{indent}    Done in {tim.dt:0.6g}s.")

    # For meshgrid use, we can store stencils as lists of linear index offsets (int32), in a ragged tensor.
    # Ragged, because points near edges or corners have a clipped stencil with fewer neighbors.
    with timer() as tim:
        if print_statistics:
            print(f"{indent}Convert stencils to tf format...")
        point_to_stencil = tf.constant(point_to_stencil, dtype=tf.int32)
        # Speed: in general, when possible, prefer `tf.ragged.stack` with a list of NumPy arrays, instead of using `tf.ragged.constant`; the first is much faster.
        #  https://github.com/tensorflow/tensorflow/issues/47853
        # But for some reason, it is faster to create `stencil_to_points` from a list of lists.
        stencil_to_points = tf.ragged.constant(stencil_to_points, dtype=tf.int32)
        # `row_splits_dtype=tf.int32` makes accesses 20% slower, better to use `int64`.
        stencils = tf.ragged.stack(stencils).with_row_splits_dtype(tf.int64)
        if print_statistics:
            print(f"{indent}    Memory usage:")
            print(f"{indent}        point_to_stencil: {sizeof_tensor(point_to_stencil)}, {point_to_stencil.dtype}")
            print(f"{indent}        stencil_to_points: {sizeof_tensor(stencil_to_points)}, {stencil_to_points.dtype}")
            print(f"{indent}        stencils: {sizeof_tensor(stencils)}, {stencils.dtype}")
    if print_statistics:
        print(f"{indent}    Done in {tim.dt:0.6g}s.")

    # Build the distance matrices.
    #
    # First apply derivative scaling for numerical stability: x' := x / xscale  ⇒  d/dx → (1 / xscale) d/dx'.
    #
    # We choose xscale to make magnitudes near 1. Similarly for yscale. We base the scaling on the grid spacing in raw coordinate space,
    # defining the furthest distance in the stencil (along each coordinate axis) in the scaled space as 1.
    #
    # We assume a uniform grid spacing. The choice of `xscale`/`yscale` depends on it, but below, we also do some specializations to a
    # uniform meshgrid that allow us to save gigabytes of VRAM compared to the general case.
    #
    # We cast to `float`, so this works also in the case where we get a scalar tensor instead of a bare scalar.
    xscale = float(X[0, 1] - X[0, 0]) * N
    yscale = float(Y[1, 0] - Y[0, 0]) * N

    # --------------------------------------------------------------------------------
    # General, topology-agnostic assembly

    # Note this routine works just as well for `c` specialized to the uniform meshgrid, and that's how we're using it here - see below.
    def cnki(dx: tf.Tensor, dy: tf.Tensor) -> tf.Tensor:
        """Compute the quadratic surrogate fitting coefficient tensor `c[n, k, i]`.

        Input: rank-2 tensors:

          `dx`: signed `x` distance
          `dy`: signed `y` distance

        where:

          dx[n, k] = signed x distance from point n to point k

        and similarly for `dy`. The indices are:

          `n`: linear index of pixel
          `k`: linear index (not offset!) of neighbor pixel

        Returns: rank-3 tensor:

          c[n, k, i]

        where `n` and `k` are as above, and:

          `i`: component of surrogate fit `(f, dx, dy, dx2, dxdy, dy2)`
        """
        if low_vram:
            # To save even more VRAM (another 50%, for a total of 75%), cast the scaled `dx` and `dy` to half precision (float16),
            # which is the dtype we store `c` in when operating in low VRAM mode.
            #
            # Although this ordering of the operations takes more VRAM than the other possibility (cast first, then scale),
            # this ordering gives us the best possible accuracy for `dx` and `dy` in the float16 representation.
            #
            # Even better (for optimal accuracy of the second-order terms) would be to compute `c` first in float32,
            # then cast the final result to float16, but that takes even more VRAM. The 6× VRAM cost, compared
            # to the current approach, is exactly what we are trying to avoid here.
            dx = tf.cast(dx / xscale, tf.float16)  # LHS: offset in scaled space
            dy = tf.cast(dy / yscale, tf.float16)
        else:
            dx = dx / xscale  # LHS: offset in scaled space
            dy = dy / yscale
        one = tf.ones_like(dx)  # for the constant term of the fit
        return tf.stack([one, dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], axis=-1)

    # Compute distance of all neighbors (in stencil) for each pixel.
    #
    # Our compute dtype is float32, so it makes no sense to keep coordinates in float64, which is the Python/NumPy default.
    # So cast them to float32 to save VRAM. (Do it first before other operations.)
    #
    # This already leads to significant VRAM savings (50%) when computing the coefficient tensor `c`.
    #
    with timer() as tim:
        if print_statistics:
            print(f"{indent}Typecast and linearize coordinate arrays...")
        X = tf.cast(X, dtype)
        Y = tf.cast(Y, dtype)
        # We'll be using linear indexing.
        X = tf.reshape(X, [-1])
        Y = tf.reshape(Y, [-1])
        if print_statistics:
            print(f"{indent}    Memory usage:")
            print(f"{indent}        X: {sizeof_tensor(X)}, {X.dtype}")
            print(f"{indent}        Y: {sizeof_tensor(Y)}, {Y.dtype}")
    print(f"{indent}    Done in {tim.dt:0.6g}s.")

    # Linear indices (not offsets!) of neighbors (in stencil) for each pixel; resolution² * (2 * N + 1)² * 4 bytes.
    # The first term is the base linear index of each pixel; the second is the linear index offset of each of its neighbors.
    #
    # `neighbors[n, k]` is the global index of neighbor `k` (local index) of pixel `n` (global index).
    #
    # We really do need this even on a meshgrid, because we need to look up actual neighbor data values when assembling the load vector `b` later.
    # Placing all the indices in a (ragged) tensor is an easy way to parallelize the lookup.
    #
    # At the cost of some compute, we can batch this at assembly time in `_assemble_b`, like we do for lookups of `c`.
    # If there is VRAM to keep a full copy of this in storage, there's no point in recomputing every time `_assemble_b` is called;
    # but if VRAM is low, then it's better to save the memory, and recompute per batch.
    if not low_vram:
        with timer() as tim:
            if print_statistics:
                print(f"{indent}Compute neighbors for each point...")
            neighbors = tf.expand_dims(tf.range(npoints), axis=-1) + tf.gather(stencils, point_to_stencil)
            if print_statistics:
                print(f"{indent}    Memory usage:")
                print(f"{indent}        neighbors: {sizeof_tensor(neighbors)}, {neighbors.dtype}")  # Spoiler: this tensor is moderately large (can be a few hundred MB).
        print(f"{indent}    Done in {tim.dt:0.6g}s.")
    else:
        neighbors = None

    # # For general topologies we need `neighbors`, and can use it as follows:
    # #   `dx[n, k]`: signed x distance of neighbor `k` (local index) from pixel `n` (global index). Similarly for `dy[n, k]`.
    # dx = tf.gather(X, neighbors) - tf.expand_dims(X, axis=-1)  # `expand_dims` explicitly, to broadcast on the correct axis
    # dy = tf.gather(Y, neighbors) - tf.expand_dims(Y, axis=-1)
    #
    # Specifically on a meshgrid, we can do this instead, and save a couple hundred MB of VRAM:
    #   `dx[stencil_id, k]`: signed x distance of neighbor `k` (local index) in stencil `stencil_id`. Similarly for `dy[stencil_id, k]`.
    with timer() as tim:
        if print_statistics:
            print(f"{indent}Compute signed distances in each stencil...")
        ps = tf.squeeze(tf.gather(stencil_to_points, [0], axis=1), axis=-1)  # for each stencil, the global index of the first point that uses that stencil
        ps_neighbors = tf.expand_dims(ps, axis=-1) + stencils  # neighbor global indices for each stencil
        dx = tf.gather(X, ps_neighbors) - tf.expand_dims(tf.gather(X, ps), axis=-1)  # signed x distances in each stencil
        dy = tf.gather(Y, ps_neighbors) - tf.expand_dims(tf.gather(Y, ps), axis=-1)
        if print_statistics:
            print(f"{indent}    Memory usage:")
            print(f"{indent}        dx: {sizeof_tensor(dx)}, {dx.dtype}")
            print(f"{indent}        dy: {sizeof_tensor(dy)}, {dy.dtype}")
    print(f"{indent}    Done in {tim.dt:0.6g}s.")

    # Finally, the surrogate fitting coefficient tensor is the following `c`.
    #
    # For general topologies, `c` takes a lot of VRAM, even at float16 - easily 1 GB at 256×256, N=20.5, p=2.0.
    #
    # Specifically on a *uniform* meshgrid, it is enough to store `c[stencil_id, k, i]`, as it only depends on the distances `dx` and `dy`.
    # This can save a gigabyte of VRAM.
    #
    # Note, however, that we will then have to indirect from pixel global index `n` to `stencil_id` when assembling the equations.
    with timer() as tim:
        if print_statistics:
            print(f"{indent}Assemble coefficient tensor...")
        c = cnki(dx, dy)
        # del dx
        # del dy
        # gc.collect()  # Attempt to clean up dangling tensors. Only important in general topologies where `c` takes a lot of VRAM.

        # Scaling factors to undo the derivative scaling,  d/dx' → d/dx.  `solve` needs this to postprocess its results.
        scale = tf.constant([1.0, xscale, yscale, xscale**2, xscale * yscale, yscale**2], dtype=dtype)
        # scale = tf.expand_dims(scale, axis=-1)  # for broadcasting; solution shape from `tf.linalg.solve` is [6, npoints]

        if print_statistics:
            print(f"{indent}    Memory usage:")
            print(f"{indent}        c: {sizeof_tensor(c)}, {c.dtype}")  # Spoiler: this tensor is huge (1 GB) in general case, small (~20 MB) in the uniform meshgrid case.
            print(f"{indent}        scale: {sizeof_tensor(scale)}, {scale.dtype}")
    print(f"{indent}    Done in {tim.dt:0.6g}s.")

    # # DEBUG: If the x and y scalings work, the range of values in `c` should be approximately [0, 1].
    # absc = tf.abs(c)
    # print(f"c[n, k, i] ∈ [{tf.reduce_min(absc):0.6g}, {tf.reduce_max(absc):0.6g}]")

    # The `A` matrices can be preassembled. They must be stored per-pixel (for easy use with linear system solver), but the size is only 6×6,
    # so at float32, we need 36 * 4 bytes * resolution² = 144 * resolution², which is only 38 MB at 512×512, and at 1024×1024, only 151 MB.
    with timer() as tim:
        if print_statistics:
            print(f"{indent}Assemble system matrix ({tf.shape(c)[0]} instances)...")  # (there are #stencils instances)
        assembled = _assemble_a(c, point_to_stencil, dtype, format, low_vram, low_vram_batch_size)
        if format == "A":
            A = assembled
            if print_statistics:
                print(f"{indent}    Memory usage:")
                print(f"{indent}        A: {sizeof_tensor(A)}, {A.dtype}")
        else:  # format == "LUp":
            LU, perm = assembled
            if print_statistics:
                print(f"{indent}    Memory usage:")
                print(f"{indent}        LU: {sizeof_tensor(LU)}, {LU.dtype}")
                print(f"{indent}        perm: {sizeof_tensor(perm)}, {perm.dtype}")
    print(f"{indent}    Done in {tim.dt:0.6g}s.")

    if format == "A":
        return (A, c, scale, point_to_stencil, stencils, neighbors), orig_interior_stencil
    return (LU, perm, c, scale, point_to_stencil, stencils, neighbors), orig_interior_stencil


# Trying to run `solve` at float16 precision on TF 2.12 gives the error:
#
# tensorflow.python.framework.errors_impl.InvalidArgumentError: No OpKernel was registered to support Op 'MatrixSolve' used by {{node MatrixSolve}} with these attrs: [T=DT_HALF, adjoint=false]
# Registered devices: [CPU, GPU]
# Registered kernels:
#   device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
#   device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
#   device='GPU'; T in [DT_COMPLEX128]
#   device='GPU'; T in [DT_COMPLEX64]
#   device='GPU'; T in [DT_DOUBLE]
#   device='GPU'; T in [DT_FLOAT]
#   device='CPU'; T in [DT_COMPLEX128]
#   device='CPU'; T in [DT_COMPLEX64]
#   device='CPU'; T in [DT_DOUBLE]
#   device='CPU'; T in [DT_FLOAT]
#
# 	 [[MatrixSolve]] [Op:__inference_solve_9642]
#
# This seems to suggest that as of TF 2.12, the XLA JIT would give us float16 precision for `tf.linalg.solve`.
# Alas, we can't JIT compile our `solve`, because `RaggedRange` is not supported by XLA, so assembling `b` fails.
# @tf.function(jit_compile=True)  # https://www.tensorflow.org/xla/tutorials/jit_compile
#
# If we split the `tf.linalg.solve` into a separate JIT-compiled wrapper function, we then get the error
# "Invalid type for triangular solve 10". The only match is `triangular_solve_thunk.cc`, one online copy here:
#   https://www.androidos.net.cn/android/10.0.0_r6/xref/external/tensorflow/tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.cc
#
# which seems to suggest that at least some versions of TF actually only support float32 and above in
# `tf.linalg.solve`, although the previous error message (upon running without JIT compilation) suggests
# otherwise. Perhaps it actually means that the *XLA compiler* can try to compile this operation for float16,
# but this does not guarantee that a kernel exists.
#
# Fortunately, a simple linear equation system solver based on the LU decomposition and two triangular solves
# isn't that complicated. See `xgesv.py`. This gives us a solver that can run also at float16, see `solve_lu`.
#
# However, upon testing this, it turned out that float32 isn't the speed bottleneck here, and the accuracy
# of float16 isn't really enough for this use case. If you want to see for yourself, the experiment is
# preserved as `solve_lu_custom`.

# NOTE: When debugging, be sure to disable the `@tf.function` also for the `solve` (or `solve_lu`, ...) routine that you're using.
#       Otherwise, we will already be in graph mode when `_assemble_b` is called.
@tf.function
def _assemble_b(c: tf.Tensor,
                point_to_stencil: tf.Tensor,
                stencils: tf.RaggedTensor,
                neighbors: typing.Optional[tf.RaggedTensor],
                z: tf.Tensor,
                low_vram: bool,
                low_vram_batch_size: int) -> tf.Tensor:
    """[internal helper] Assemble the load vector.

    The formula is::

      b[n,i] = ∑k( z[neighbors[n,k]] * c[n,k,i] )

    For the parameters, see `solve`, except:

    `z`: *Linearized* (rank-1, linear numbering of DOFs) representation of the original `z` data.

         `solve` does this reshaping automatically, so this is mainly important for
         debugging and unit-testing of `_assemble_b` itself.

    Returns a `tf.Tensor` of shape [npoints, 6], containing the pixelwise loads.
    """
    # Ensure numeric value even for symbolic tensor input - we may be called from inside a `@tf.function`.
    npoints = tf.get_static_value(tf.shape(z), partial=True)[0]  # NOTE: Reshaped, linearly indexed `z`.
    low_vram_batch_size = min(npoints, low_vram_batch_size)  # can't have a batch larger than there is data

    # Save some VRAM (< 100 MB) by letting `neighbors_split` fall out of scope as soon as it's no longer needed.
    @tf.function  # we're already inside a `@tf.function`, so this is just to document intent.
    def _get_zgnk(start: tf.Tensor, stop: tf.Tensor) -> tf.Tensor:  # `start` and `stop` are wrapped scalars
        # The first term is the base linear index of each pixel; the second is the linear index offset of each of its neighbors.
        neighbors_split = tf.expand_dims(tf.range(start, stop), axis=-1) + tf.gather(stencils, point_to_stencil[start:stop])  # [#split, #k], ragged in k
        zgnk_split = tf.gather(z, neighbors_split, name="gather_neighbors")  # [#n] -> [#split, #k], ragged in k
        return zgnk_split

    @tf.function
    def _assemble_bi(i: tf.Tensor,  # wrapped scalar
                     c_expanded: tf.Tensor,
                     zgnk: tf.Tensor) -> tf.Tensor:
        # We upcast after slicing to save VRAM.
        ci = tf.cast(c_expanded[:, :, i], tf.float32, name="cast_to_float32")  # -> [#split, #k] (batched) or [#n, #k] (not batched)
        bi = tf.reduce_sum(zgnk * ci, axis=1, name="assemble_bi")  # [#split, #k] -> [#split] (batched) or [#n, #k] -> [#n] (not batched)
        return bi

    if not low_vram:  # `neighbors` is precomputed. No batching.
        assert neighbors is not None
        # The `RaggedSplitsToSegmentIds`, used internally by TF inside the `reduce_sum` costs a lot of VRAM, since it indexes using `int64`,
        # and we're handling a lot of tensor elements here. E.g. at 256×256, #n = 65536, and with N=13, p=2.0, #k ~ 500, we have about 30M points;
        # so with an int64 for each, we're talking ~300 MB just for the indexing. When the chosen settings are already pushing against the VRAM limit,
        # this is the straw that breaks the camel's back.
        #
        # When enough VRAM is available, this is the simplest way to do the assembly.
        #
        # We need to be slightly careful with loops, to avoid unrolling them in the graph (which slows down tracing, so program startup;
        # and makes the graph much larger than it needs to be). The computations are rather data-heavy, so unrolling shouldn't give much
        # of a performance boost anyway.
        # https://www.tensorflow.org/guide/function#loops
        # https://www.tensorflow.org/guide/function#accumulating_values_in_a_loop
        rows = tf.TensorArray(z.dtype, size=6)
        zgnk = tf.gather(z, neighbors, name="gather_neighbors")  # -> [#n, #k], ragged in k
        c_expanded = tf.gather(c, point_to_stencil, axis=0, name="expand_c")  # [#nstencils, #k, #rows] -> [#n, #k, #rows]; this can take GBs of VRAM!
        for i in tf.range(6):
            bi = _assemble_bi(i, c_expanded, zgnk)  # -> [#n]
            rows = rows.write(i, tf.cast(bi, z.dtype, name="cast_to_dtype"))
        rows = rows.stack()  # -> [#rows, #n]
        b = tf.transpose(rows, [1, 0])  # -> [#n, #rows]
    else:  # `neighbors is None`, to be computed on-the-fly. Batched assembly.
        assert neighbors is None
        # In low VRAM mode, we assemble `b` in batches, splitting over the meshgrid points.
        n_batches = math.ceil(npoints / low_vram_batch_size)
        batches = [[j, j * low_vram_batch_size, (j + 1) * low_vram_batch_size] for j in range(n_batches)]  # [[batch_index, start, stop], ...]
        batches[-1][-1] = npoints  # Set the final `stop` value to include all remaining tensor elements in the final batch.
        batches = tf.constant(batches)  # important: convert batch metadata to `tf.Tensor`...
        rows_by_batch = tf.TensorArray(z.dtype, size=n_batches)
        for batch_metadata in batches:  # ...so that TF can compile this `for element in tensor` into a loop node in the graph, avoiding unrolling.
            # Must unpack metadata manually, since iterating over a symbolic `tf.Tensor` is not allowed.
            batch_index = batch_metadata[0]
            start = batch_metadata[1]
            stop = batch_metadata[2]
            zgnk_split = _get_zgnk(start, stop)
            c_split_expanded = tf.gather(c, point_to_stencil[start:stop], axis=0, name="expand_c")  # [#nstencils, #k, #rows] -> [#split, #k, #rows]
            rows = tf.TensorArray(z.dtype, size=6)
            for i in tf.range(6):
                bi_split = _assemble_bi(i, c_split_expanded, zgnk_split)  # -> [#split]
                rows = rows.write(i, tf.cast(bi_split, z.dtype, name="cast_to_dtype"))
            rows = rows.stack()  # [#rows, #split]
            rows_by_batch = rows_by_batch.write(batch_index, rows)
        rows_by_batch = rows_by_batch.stack()  # [#batches, #rows, #split]
        rows_by_batch = tf.transpose(rows_by_batch, [0, 2, 1])  # -> [#batches, #split, #rows]
        b = tf.reshape(rows_by_batch, shape=[n_batches * low_vram_batch_size, 6])  # unsplit -> [#n, #rows]
    return b

@tf.function
def solve(a: tf.Tensor,
          c: tf.Tensor,
          scale: tf.Tensor,
          stencils: tf.RaggedTensor,
          point_to_stencil: tf.Tensor,
          neighbors: typing.Optional[tf.RaggedTensor],
          z: tf.Tensor,
          low_vram: bool = True,
          low_vram_batch_size: int = 8192) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    This uses `tf.linalg.solve`, and is the recommended algorithm.

    `a`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="A"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batch_size`: If `low_vram=True`, this is the batch size for assembling the load vector.
                           Decrease this to trade off speed for lower VRAM usage.

                           If `low_vram=False`, this parameter is ignored.

    This function runs completely on the GPU, and is differentiable w.r.t. `z`, so it can be used
    e.g. inside a loss function for a neural network that predicts `z` (if such a loss function
    happens to need the spatial derivatives of `z`, as estimated from image data).

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, dx2, dxdy, dy2, in that order.
    """
    shape = tf.shape(z)
    z = tf.cast(z, tf.float32)
    z = tf.reshape(z, [-1])

    zmax = tf.reduce_max(tf.abs(z), name="find_zmax")
    z = z / zmax  # -> [-1, 1]

    b = _assemble_b(c, point_to_stencil, stencils, neighbors, z, low_vram, low_vram_batch_size)  # -> [#n, #rows]

    b = tf.expand_dims(b, axis=-1)  # -> [#n, #rows, 1]  (in this variant of the algorithm, we have just one RHS for each LHS matrix)
    sol = tf.linalg.solve(a, b, name="solve_system")  # -> [#n, #rows, 1]
    # print(tf.shape(sol))  # [#n, #rows, 1]
    # print(tf.math.reduce_max(abs(tf.matmul(a, sol) - b)))  # DEBUG: yes, the solutions are correct.
    sol = tf.squeeze(sol, axis=-1)  # -> [#n, #rows]

    sol = sol / scale  # return derivatives from scaled x, y (as set up by `prepare`) to raw x, y
    sol = zmax * sol  # return from scaled z to raw z

    sol = tf.transpose(sol, [1, 0])  # -> [#rows, #n]
    return tf.reshape(sol, (6, int(shape[0]), int(shape[1])))  # -> [#rows, ny, nx]

@tf.function
def solve_lu(lu: tf.Tensor,
             p: tf.Tensor,
             c: tf.Tensor,
             scale: tf.Tensor,
             point_to_stencil: tf.Tensor,
             stencils: tf.RaggedTensor,
             neighbors: typing.Optional[tf.RaggedTensor],
             z: tf.Tensor,
             low_vram: bool = True,
             low_vram_batch_size: int = 8192) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    This uses a TensorFlow's LU solver kernel.

    `lu`, `p`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="LUp"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batch_size`: If `low_vram=True`, this is the batch size for assembling the load vector.
                           Decrease this to trade off speed for lower VRAM usage.

                           If `low_vram=False`, this parameter is ignored.

    This function runs completely on the GPU, and is differentiable w.r.t. `z`, so it can be used
    e.g. inside a loss function for a neural network that predicts `z` (if such a loss function
    happens to need the spatial derivatives of `z`, as estimated from image data).

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, dx2, dxdy, dy2, in that order.
    """
    shape = tf.shape(z)
    z = tf.cast(z, tf.float32)
    z = tf.reshape(z, [-1])

    zmax = tf.reduce_max(tf.abs(z), name="find_zmax")
    z = z / zmax  # -> [-1, 1]

    b = _assemble_b(c, point_to_stencil, stencils, neighbors, z, low_vram, low_vram_batch_size)  # -> [#n, #rows]

    b = tf.expand_dims(b, axis=-1)  # -> [#n, #rows, 1]  (in this variant of the algorithm, we have just one RHS for each LHS matrix)
    sol = tf.linalg.lu_solve(lu, p, b, name="solve_system_lu")  # [#n, #rows, 1]
    sol = tf.squeeze(sol, axis=-1)  # -> [#n, #rows]

    sol = sol / scale  # return derivatives from scaled x, y (as set up by `prepare`) to raw x, y
    sol = zmax * sol  # return from scaled z to raw z

    sol = tf.transpose(sol, [1, 0])  # -> [#rows, #n]
    return tf.reshape(sol, (6, int(shape[0]), int(shape[1])))  # -> [#rows, ny, nx]

def solve_lu_custom(lu: tf.Tensor,
                    p: tf.Tensor,
                    c: tf.Tensor,
                    scale: tf.Tensor,
                    point_to_stencil: tf.Tensor,
                    stencils: tf.RaggedTensor,
                    neighbors: typing.Optional[tf.RaggedTensor],
                    z: tf.Tensor,
                    low_vram: bool = True,
                    low_vram_batch_size: int = 8192) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    This uses a custom LU solver kernel, which can run also at float16 (although generally,
    that is not very useful; this problem tends to need float32 for acceptable accuracy).

    `lu`, `p`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="LUp"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batch_size`: If `low_vram=True`, this is the batch size for assembling the load vector.
                           Decrease this to trade off speed for lower VRAM usage.

                           If `low_vram=False`, this parameter is ignored.

    This function runs completely on the GPU, and is differentiable w.r.t. `z`, so it can be used
    e.g. inside a loss function for a neural network that predicts `z` (if such a loss function
    happens to need the spatial derivatives of `z`, as estimated from image data).

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, dx2, dxdy, dy2, in that order.
    """
    shape = tf.shape(lu)
    batch = int(shape[0])
    n = int(shape[1])
    x = tf.Variable(tf.zeros([batch, n], dtype=lu.dtype), name="x")  # cannot be allocated inside @tf.function
    return _solve_lu_custom_kernel(lu, p, c, scale, point_to_stencil, stencils, neighbors, z, low_vram, low_vram_batch_size, x)

@tf.function
def _solve_lu_custom_kernel(lu: tf.Tensor,
                            p: tf.Tensor,
                            c: tf.Tensor,
                            scale: tf.Tensor,
                            point_to_stencil: tf.Tensor,
                            stencils: tf.RaggedTensor,
                            neighbors: typing.Optional[tf.RaggedTensor],
                            z: tf.Tensor,
                            low_vram: bool,
                            low_vram_batch_size: int,
                            x: tf.Variable) -> tf.Tensor:
    """[internal helper] The actual computational kernel for `solve_lu_custom`.

    `lu`, `p`, `c`, `scale`, `neighbors`, `z`: passed through from API wrapper `solve_lu`.
    `x`: work space variable, [batch, n]. Will be written to.

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, dx2, dxdy, dy2, in that order.
    """
    shape = tf.shape(z)
    z = tf.cast(z, tf.float32)
    z = tf.reshape(z, [-1])

    zmax = tf.reduce_max(tf.abs(z), name="find_zmax")
    z = z / zmax  # -> [-1, 1]

    b = _assemble_b(c, point_to_stencil, stencils, neighbors, z, low_vram, low_vram_batch_size)  # -> [#n, #rows]

    # We must call the kernel directly, because we are inside @tf.function;
    # the API wrapper would try to allocate its own `x`, which we can't do here.
    lusolve(lu, p, b, x)  # our custom kernel, writes solution into `x`
    sol = x

    sol = sol / scale  # return derivatives from scaled x, y (as set up by `prepare`) to raw x, y
    sol = zmax * sol  # return from scaled z to raw z

    sol = tf.transpose(sol, [1, 0])  # -> [#rows, #n]
    return tf.reshape(sol, (6, int(shape[0]), int(shape[1])))  # -> [#rows, ny, nx]


# ----------------------------------------------------------------------------------------
# !!! Everything below this point is DEPRECATED, and preserved for documentation only. !!!
# ----------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Basic API. Slower, not as accurate in corners.
#
# See `wlsqm.pdf` in the `python-wlsqm` docs for details on the algorithm.

# helper function
def make_interior_stencil(N: int) -> np.array:
    """Return an array of integer offset pairs for a square-shaped stencil of N×N points.

    `N`: neighborhood size parameter (how many grid spacings on each axis)

    Return value is a rank-2 np-array of shape [n_neighbors, 2].
    """
    if N is None or N < 1:
        raise ValueError(f"Must specify N ≥ 1, got {N}")
    neighbors = [[iy, ix] for iy in range(-N, N + 1)
                          for ix in range(-N, N + 1)]  # if not (iy == 0 and ix == 0) ...but including the center point does no harm.
    neighbors = np.array(neighbors, dtype=int)
    return neighbors


def differentiate(N: typing.Optional[int],
                  X: typing.Union[np.array, tf.Tensor],
                  Y: typing.Union[np.array, tf.Tensor],
                  Z: typing.Union[np.array, tf.Tensor],
                  *,
                  padding: str,
                  stencil: np.array = None) -> tf.Tensor:
    """[kernel] Fit a 2nd order surrogate polynomial to data values on a meshgrid, to estimate derivatives.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    Note the distance matrix `A` (generated automatically) is 5×5 regardless of `N`, but for large `N`,
    assembly takes longer because there are more contributions to each matrix element.

    `N`: neighborhood size parameter (how many grid spacings on each axis)
    `X`, `Y`, `Z`: data in meshgrid format for x, y, and function value, respectively

    `padding`: similar to convolution operations, one of:
        "VALID": Operate in the interior only. This chops off `N` points at the edges on each axis.
        "SAME": Preserve data tensor dimensions. Automatically use local extrapolation to estimate
                `X`, `Y`, and `Z` outside the edges.

    `stencil`: Optional: rank-2 np.array of integer offsets `[[Δx0, Δy0], [Δx1, Δy1], ...]`, in grid units.
               This explicitly specifies which grid points to include to estimate the local model.

               When specified:
                 - Only the `padding="VALID"` mode is available (if you need edges too, consider
                   the higher level API `hifi_differentiate` or `hifier_differentiate`),
                 - `N` is ignored,
                 - This custom `stencil` is used instead of automatically constructing a centered stencil
                   based on the value of `N`.
               The position of the valid part is automatically determined based on the stencil.

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    dx, dy, dx2, dxdy, dy2, in that order.
    """
    if padding.upper() not in ("VALID", "SAME"):
        raise ValueError(f"Invalid padding '{padding}'; valid choices: 'VALID', 'SAME'")
    if stencil is None and N is None:
        raise ValueError("Must specify exactly one of `stencil` or `N`; neither was given.")
    if stencil is not None and N is not None:
        raise ValueError("Cannot specify both a custom `stencil` and `N`. When using a custom `stencil`, please call with `N=None`.")
    if stencil is not None and padding.upper() == "SAME":
        raise ValueError("Cannot use `padding='SAME'` with a custom `stencil`. Please use `padding='VALID'`.")
    if padding.upper() == "SAME":
        assert N is not None
        X = pad_linear_2d(N, X)
        Y = pad_linear_2d(N, Y)
        Z = pad_quadratic_2d(N, Z)
    for name, tensor in (("X", X), ("Y", Y), ("Z", Z)):
        shape = tf.shape(tensor)
        if len(shape) != 2:
            raise ValueError(f"Expected `{name}` to be a rank-2 tensor; got rank {len(shape)}")
        if not (int(shape[0]) >= 2 and int(shape[1]) >= 2):
            raise ValueError(f"Expected `{name}` to be at least of size [2 2]; got {shape}")

    # Generic offset distance stencil for all neighbors.
    neighbors = stencil if stencil is not None else make_interior_stencil(N)  # [#k, 2]
    min_yoffs = np.min(neighbors[:, 0])
    max_yoffs = np.max(neighbors[:, 0])
    min_xoffs = np.min(neighbors[:, 1])
    max_xoffs = np.max(neighbors[:, 1])
    max_abs_yoffs = max(abs(min_yoffs), abs(max_yoffs))
    max_abs_xoffs = max(abs(min_xoffs), abs(max_xoffs))

    # Derivative scaling for numerical stability: x' := x / xscale  ⇒  d/dx → (1 / xscale) d/dx'.
    #
    # We choose xscale to make magnitudes near 1. Similarly for y. We base the scaling on the grid spacing in raw coordinate space,
    # defining the furthest distance in the stencil (along each coordinate axis) in the scaled space as 1.
    #
    # We need to cast to `float` in case `tf` decides to give us a scalar tensor instead of a bare scalar. This happens when `X` and `Y` are padded.
    xscale = float(X[0, 1] - X[0, 0]) * max_abs_xoffs
    yscale = float(Y[1, 0] - Y[0, 0]) * max_abs_yoffs

    # Scale the data values to [-1, 1], too.
    zscale = tf.reduce_max(tf.abs(Z)).numpy()
    Z = Z / zscale

    def cki(dx, dy):
        """Compute the `c[k, i]` coefficients for surrogate fitting.

        Essentially, the quadratic surrogate is based on the Taylor series of our function `f`::

          f(xk, yk) ≈ f(xi, yi) + ∂f/∂x dx + ∂f/∂y dy + ∂²f/∂x² (1/2 dx²) + ∂²f/∂x∂y dx dy + ∂²f/∂y² (1/2 dy²)
                   =: f(xi, yi) + ∂f/∂x c[k,0] + ∂f/∂y c[k,1] + ∂²f/∂x² c[k,2] + ∂²f/∂x∂y c[k,3] + ∂²f/∂y² c[k,4]

        where the c[k,i] are known geometric factors. Given a neighborhood around a given center point (xi, yi)
        with 6 or more pixels (xk, yk, fk), we can write a linear equation system that yields approximate
        derivatives of the quadratic surrogate at the center point (xi, yi).

        Because we are solving for 5 coefficients, we must use at least 6 neighbors to make the fitting problem
        overdetermined (hence "least squares meshfree method"). Note that we are not actually solving an
        overdetermined *linear equation system*. Although our *problem* is overdetermined, it turns out that
        we can solve this problem using an appropriately constructed linear equation system with a square matrix.
        See the `python-wlsqm` docs for details.

        Also, the surrogate is *not* an exact truncation of the Taylor series of `f`, in the following sense:

        If we were to later fit a higher-order surrogate, and compare the result to the quadratic surrogate,
        also all the lower-order approximate derivatives would have slightly different values (even though the
        algorithm is completely deterministic). This is unlike in a Taylor series, where increasing the order
        adds more terms, but does not change the values of the ones already computed. The surrogate is
        least-squares optimal, at the given order (and *only* at that order), absorbing also the truncation
        error into the coefficients.

        `dx`, `dy`: offset distance in raw coordinate space. Either:
            - `float`, for a single pair of pixels
            - rank-1 `np.array`, for a batch of pixel pairs

        Return value:
            For float input: `c[i]`, rank-1 `np.array` of shape `(5,)`
            For array input with `k` elements: `c[k, i]`, rank-2 `np.array` of shape `(n, 5)`
        """
        dx = dx / xscale  # LHS: offset in scaled space (where the furthest distance in the stencil is 1)
        dy = dy / yscale
        return np.array([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]).T

    # Since we have a uniform grid in this application, the distance matrix of neighbors for each point is the same,
    # so we need to assemble only one.

    iy, ix = -min_yoffs, -min_xoffs  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    # # This is what we want to do, but this crashes (invalid slice) if we hand it a padded `X` (which becomes a `tf.Tensor`):
    # dx = X[iy + neighbors[:, 0], ix + neighbors[:, 1]] - X[iy, ix]  # [#k]
    # dy = Y[iy + neighbors[:, 0], ix + neighbors[:, 1]] - Y[iy, ix]  # [#k]
    # So let's do it the `tf` way instead:
    indices = tf.constant(list(zip(iy + neighbors[:, 0], ix + neighbors[:, 1])))
    dx = tf.gather_nd(X, indices) - X[iy, ix]  # [#k]
    dy = tf.gather_nd(Y, indices) - Y[iy, ix]  # [#k]

    # Tensor viewpoint. Define the indices as follows:
    #   `n`: pixel in batch,
    #   `k`: neighbor,
    #   `i`: row of MLS equation system "A x = b"
    #   `j`: column of MLS equation system "A x = b"
    # and let `f[n] = f(x[n], y[n])`.
    #
    # Then, in the general case, the MLS equation systems for the batch are given by:
    #   A[n,i,j] = ∑k( c[n,k,i] * c[n,k,j] )
    #   b[n,i] = ∑k( (f[g[n,k]] - f[n]) * c[n,k,i] )
    #
    # where `g[n,k]` is the (global) pixel index, of neighbor `k` of pixel `n`.
    #
    # On a uniform grid, c[n1,k,i] = c[n2,k,i] =: c[k,i] for any n1, n2, so this simplifies to:
    #   A[i,j] = ∑k( c[k,i] * c[k,j] )
    #   b[n,i] = ∑k( (f[g[n,k]] - f[n]) * c[k,i] )
    #
    # In practice we still have to chop the edges, which modifies the pixel indexing slightly.
    # Since we use a single stencil, we can form the equations for the interior part only, whereas
    # for the neighbor data values we need to use the full original data.

    # Assemble `A`:
    c = cki(dx, dy)  # [#k, 5]
    A = tf.einsum("ki,kj->ij", c, c)  # A[i,j] = ∑k( c[k,i] * c[k,j] )

    # # In other words:
    # A = np.zeros((5, 5))
    # iy, ix = N, N
    # for offset_y, offset_x in neighbors:
    #     dx = X[iy + offset_y, ix + offset_x] - X[iy, ix]
    #     dy = Y[iy + offset_y, ix + offset_x] - Y[iy, ix]
    #     c = cki(dx, dy)
    #     for j in range(5):
    #         for n in range(5):
    #             A[j, n] += c[j] * c[n]
    # A = tf.constant(A)

    # Form the right-hand side for each pixel. This is the only part that depends on the data values f.
    #
    # As per the above summary, we need to compute:
    #   b[n,i] = ∑k( (f[g[n,k]] - f[n]) * c[k,i] )
    #
    # Let
    #   df[n,k] := f[g[n,k]] - f[n]
    # Then
    #   b[n,i] = ∑k( df[n,k] * c[k,i] )
    # which can be implemented as:
    #   b = tf.einsum("nk,ki->ni", df, c)
    #
    # The tricky part is computing the index sets for df. First, we index the function value data linearly:
    f = tf.reshape(Z, [-1])

    # Then determine the multi-indices for the interior points:
    npoints = tf.reduce_prod(tf.shape(X))
    all_multi_to_linear = tf.reshape(tf.range(npoints), tf.shape(X))  # e.g. [0, 1, 2, 3, 4, 5, ...] -> [[0, 1, 2], [3, 4, 5], ...]; C storage order assumed

    ystart = -min_yoffs
    ystop = -max_yoffs if max_yoffs else None
    xstart = -min_xoffs
    xstop = -max_xoffs if max_xoffs else None
    interior_multi_to_linear = all_multi_to_linear[ystart:ystop, xstart:xstop]  # take the valid part

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior pixel
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior pixel

    # Now we can evaluate df, and finally b.
    df = tf.gather(f, gnk) - tf.gather(f, n)  # [n_interior_points, #k]
    # bs = tf.einsum("nk,ki->ni", df, c)  # This would be clearer...
    bs = tf.einsum("nk,ki->in", df, c)  # ...but this is the ordering `tf.linalg.solve` wants (components on axis 0, batch on axis 1).
    # (This ordering is when solving many RHS for the same LHS matrix.)

    # # In other words:
    # ny, nx = np.shape(X)
    # bs = []
    # for iy in range(N, ny - N):
    #     for ix in range(N, nx - N):
    #         b = np.zeros((5,))
    #         for offset_y, offset_x in neighbors:
    #             dx = X[iy + offset_y, ix + offset_x] - X[iy, ix]
    #             dy = Y[iy + offset_y, ix + offset_x] - Y[iy, ix]
    #             dz = Z[iy + offset_y, ix + offset_x] - Z[iy, ix]  # function value delta
    #             c = cki(dx, dy)
    #             for j in range(5):
    #                 b[j] += dz * c[j]
    #         bs.append(b)
    # bs = tf.stack(bs, axis=1)

    # The solution of the linear systems (one per pixel) yields the jacobian and hessian of the surrogate.
    df = tf.linalg.solve(A, bs)  # [5, n_interior_points]

    # Undo the derivative scaling,  d/dx' → d/dx
    scale = tf.constant([xscale, yscale, xscale**2, xscale * yscale, yscale**2], dtype=df.dtype)
    scale = tf.expand_dims(scale, axis=-1)  # for broadcasting
    df = df / scale

    df = tf.reshape(df, (5, *tf.shape(interior_multi_to_linear)))
    return df * zscale


# --------------------------------------------------------------------------------
# Fit also function values.
#
# See `wlsqm_gen.pdf` in the `python-wlsqm` docs for details on the algorithm.

def fit_quadratic(N: typing.Optional[int],
                  X: typing.Union[np.array, tf.Tensor],
                  Y: typing.Union[np.array, tf.Tensor],
                  Z: typing.Union[np.array, tf.Tensor],
                  *,
                  padding: str,
                  stencil: np.array = None) -> tf.Tensor:
    """[kernel] Like `differentiate`, but fit function values too.

    Each pixel is associated with a local quadratic model (in an overlapping patch of pixels).

    As well as estimating derivatives, this can be used as a local least squares denoiser.

    Note the distance matrix `A` (generated automatically) is 6×6 regardless of `N`, but for large `N`,
    assembly takes longer because there are more contributions to each matrix element.

    `N`: neighborhood size parameter (how many grid spacings on each axis)
    `X`, `Y`, `Z`: data in meshgrid format for x, y, and function value, respectively

    `padding`: similar to convolution operations, one of:
        "VALID": Operate in the interior only. This chops off `N` points at the edges on each axis.
        "SAME": Preserve data tensor dimensions. Automatically use local extrapolation to estimate
                `X`, `Y`, and `Z` outside the edges.

    `stencil`: Optional: rank-2 np.array of integer offsets `[[Δx0, Δy0], [Δx1, Δy1], ...]`, in grid units.
               This explicitly specifies which grid points to include to estimate the local model.

               When specified:
                 - Only the `padding="VALID"` mode is available (if you need edges too, consider
                   the higher level API `hifi_differentiate` or `hifier_differentiate`),
                 - `N` is ignored,
                 - This custom `stencil` is used instead of automatically constructing a centered stencil
                   based on the value of `N`.
               The position of the valid part is automatically determined based on the stencil.

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, dx2, dxdy, dy2, in that order.
    """
    if padding.upper() not in ("VALID", "SAME"):
        raise ValueError(f"Invalid padding '{padding}'; valid choices: 'VALID', 'SAME'")
    if stencil is None and N is None:
        raise ValueError("Must specify exactly one of `stencil` or `N`; neither was given.")
    if stencil is not None and N is not None:
        raise ValueError("Cannot specify both a custom `stencil` and `N`. When using a custom `stencil`, please call with `N=None`.")
    if stencil is not None and padding.upper() == "SAME":
        raise ValueError("Cannot use `padding='SAME'` with a custom `stencil`. Please use `padding='VALID'`.")
    if padding.upper() == "SAME":
        assert N is not None
        X = pad_linear_2d(N, X)
        Y = pad_linear_2d(N, Y)
        Z = pad_quadratic_2d(N, Z)
    for name, tensor in (("X", X), ("Y", Y), ("Z", Z)):
        shape = tf.shape(tensor)
        if len(shape) != 2:
            raise ValueError(f"Expected `{name}` to be a rank-2 tensor; got rank {len(shape)}")
        if not (int(shape[0]) >= 2 and int(shape[1]) >= 2):
            raise ValueError(f"Expected `{name}` to be at least of size [2 2]; got {repr(shape)}")

    neighbors = stencil if stencil is not None else make_interior_stencil(N)  # [#k, 2]
    min_yoffs = np.min(neighbors[:, 0])
    max_yoffs = np.max(neighbors[:, 0])
    min_xoffs = np.min(neighbors[:, 1])
    max_xoffs = np.max(neighbors[:, 1])
    max_abs_yoffs = max(abs(min_yoffs), abs(max_yoffs))
    max_abs_xoffs = max(abs(min_xoffs), abs(max_xoffs))

    xscale = float(X[0, 1] - X[0, 0]) * max_abs_xoffs
    yscale = float(Y[1, 0] - Y[0, 0]) * max_abs_yoffs

    zscale = tf.reduce_max(tf.abs(Z)).numpy()
    Z = Z / zscale

    def cki(dx, dy):
        dx = dx / xscale
        dy = dy / yscale
        one = tf.ones_like(dx)  # for the constant term of the fit
        return np.array([one, dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]).T

    iy, ix = -min_yoffs, -min_xoffs  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    indices = tf.constant(list(zip(iy + neighbors[:, 0], ix + neighbors[:, 1])))
    dx = tf.gather_nd(X, indices) - X[iy, ix]  # [#k]
    dy = tf.gather_nd(Y, indices) - Y[iy, ix]  # [#k]

    # Definitions as before. In the general case, the MLS equation systems for the batch are now given by:
    #   A[n,i,j] = ∑k( c[n,k,i] * c[n,k,j] )
    #   b[n,i] = ∑k( f[g[n,k]] * c[n,k,i] )
    #
    # where `g[n,k]` is the (global) pixel index, of neighbor `k` of pixel `n`.
    #
    # On a uniform grid, c[n1,k,i] = c[n2,k,i] =: c[k,i] for any n1, n2, so this simplifies to:
    #   A[i,j] = ∑k( c[k,i] * c[k,j] )
    #   b[n,i] = ∑k( f[g[n,k]] * c[k,i] )
    #
    # Note we now have a constant term in the fit (there's one more value for the index `k`),
    # and `f[n]` is no longer needed in `b`. We essentially take the `f[g[n,k]]` as given
    # (we will least-squares over them anyway, so it doesn't matter if they're noisy),
    # and treat `f[n]` as missing. We then attempt to fit a constant term as an approximation
    # for `f[n]`. Details in `wlsqm_gen.pdf`.
    #
    # (Note that the function value at the center point can actually be used, simply by including
    #  the center point `[0, 0]` in the stencil. Then it's treated as one of the `f[g[n,k]]`,
    #  just like the actual neighbors. Only the constant term is nonzero at the center.)

    # Assemble `A`:
    c = cki(dx, dy)  # [#k, 6]
    A = tf.einsum("ki,kj->ij", c, c)  # A[i,j] = ∑k( c[k,i] * c[k,j] )

    # Form the right-hand side for each pixel. This is the only part that depends on the data values f.
    #
    # As per the above summary, we need to compute:
    #   b[n,i] = ∑k( f[g[n,k]] * c[k,i] )
    #
    # which can be implemented as:
    #   b = tf.einsum("nk,ki->ni", fgnk, c)
    #
    # The tricky part is computing the index sets for df. First, we index the function value data linearly:
    f = tf.reshape(Z, [-1])

    # Then determine the multi-indices for the interior points:
    npoints = tf.reduce_prod(tf.shape(X))
    all_multi_to_linear = tf.reshape(tf.range(npoints), tf.shape(X))  # e.g. [0, 1, 2, 3, 4, 5, ...] -> [[0, 1, 2], [3, 4, 5], ...]; C storage order assumed

    ystart = -min_yoffs
    ystop = -max_yoffs if max_yoffs else None
    xstart = -min_xoffs
    xstop = -max_xoffs if max_xoffs else None
    interior_multi_to_linear = all_multi_to_linear[ystart:ystop, xstart:xstop]  # take the valid part

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior pixel
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior pixel

    # Now we can evaluate df, and finally b.
    fgnk = tf.gather(f, gnk)  # [n_interior_points, #k]
    bs = tf.einsum("nk,ki->in", fgnk, c)

    df = tf.linalg.solve(A, bs)  # [6, n_interior_points]

    # Undo the derivative scaling,  d/dx' → d/dx
    scale = tf.constant([1.0, xscale, yscale, xscale**2, xscale * yscale, yscale**2], dtype=df.dtype)
    scale = tf.expand_dims(scale, axis=-1)  # for broadcasting
    df = df / scale

    df = tf.reshape(df, (6, *tf.shape(interior_multi_to_linear)))
    return df * zscale


def fit_linear(N: typing.Optional[int],
               X: typing.Union[np.array, tf.Tensor],
               Y: typing.Union[np.array, tf.Tensor],
               Z: typing.Union[np.array, tf.Tensor],
               *,
               padding: str,
               stencil: np.array = None) -> tf.Tensor:
    """[kernel] Like `fit_quadratic`, but fit function values and first derivatives only.

    Each pixel is associated with a local linear model (in an overlapping patch of pixels).

    This can be used as a local least squares denoiser.

    Return value is a rank-3 tensor of shape `[channels, ny, nx]`, where `channels` are
    f, dx, dy, in that order.
    """
    if padding.upper() not in ("VALID", "SAME"):
        raise ValueError(f"Invalid padding '{padding}'; valid choices: 'VALID', 'SAME'")
    if stencil is None and N is None:
        raise ValueError("Must specify exactly one of `stencil` or `N`; neither was given.")
    if stencil is not None and N is not None:
        raise ValueError("Cannot specify both a custom `stencil` and `N`. When using a custom `stencil`, please call with `N=None`.")
    if stencil is not None and padding.upper() == "SAME":
        raise ValueError("Cannot use `padding='SAME'` with a custom `stencil`. Please use `padding='VALID'`.")
    if padding.upper() == "SAME":
        assert N is not None
        X = pad_linear_2d(N, X)
        Y = pad_linear_2d(N, Y)
        Z = pad_quadratic_2d(N, Z)
    for name, tensor in (("X", X), ("Y", Y), ("Z", Z)):
        shape = tf.shape(tensor)
        if len(shape) != 2:
            raise ValueError(f"Expected `{name}` to be a rank-2 tensor; got rank {len(shape)}")
        if not (int(shape[0]) >= 2 and int(shape[1]) >= 2):
            raise ValueError(f"Expected `{name}` to be at least of size [2 2]; got {shape}")

    neighbors = stencil if stencil is not None else make_interior_stencil(N)  # [#k, 2]
    min_yoffs = np.min(neighbors[:, 0])
    max_yoffs = np.max(neighbors[:, 0])
    min_xoffs = np.min(neighbors[:, 1])
    max_xoffs = np.max(neighbors[:, 1])
    max_abs_yoffs = max(abs(min_yoffs), abs(max_yoffs))
    max_abs_xoffs = max(abs(min_xoffs), abs(max_xoffs))

    xscale = float(X[0, 1] - X[0, 0]) * max_abs_xoffs
    yscale = float(Y[1, 0] - Y[0, 0]) * max_abs_yoffs

    zscale = tf.reduce_max(tf.abs(Z)).numpy()
    Z = Z / zscale

    def cki(dx, dy):
        dx = dx / xscale
        dy = dy / yscale
        one = tf.ones_like(dx)  # for the constant term of the fit
        return np.array([one, dx, dy]).T

    iy, ix = -min_yoffs, -min_xoffs  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    indices = tf.constant(list(zip(iy + neighbors[:, 0], ix + neighbors[:, 1])))
    dx = tf.gather_nd(X, indices) - X[iy, ix]  # [#k]
    dy = tf.gather_nd(Y, indices) - Y[iy, ix]  # [#k]

    # Assemble `A`:
    c = cki(dx, dy)  # [#k, 3]
    A = tf.einsum("ki,kj->ij", c, c)  # A[i,j] = ∑k( c[k,i] * c[k,j] )

    # Form the right-hand side for each point.
    #   b[n,i] = ∑k( f[g[n,k]] * c[k,i] )
    f = tf.reshape(Z, [-1])

    npoints = tf.reduce_prod(tf.shape(X))
    all_multi_to_linear = tf.reshape(tf.range(npoints), tf.shape(X))  # e.g. [0, 1, 2, 3, 4, 5, ...] -> [[0, 1, 2], [3, 4, 5], ...]; C storage order assumed

    ystart = -min_yoffs
    ystop = -max_yoffs if max_yoffs else None
    xstart = -min_xoffs
    xstop = -max_xoffs if max_xoffs else None
    interior_multi_to_linear = all_multi_to_linear[ystart:ystop, xstart:xstop]  # take the valid part

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior pixel
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior pixel

    # Now we can evaluate df, and finally b.
    fgnk = tf.gather(f, gnk)  # [n_interior_points, #k]
    bs = tf.einsum("nk,ki->in", fgnk, c)

    df = tf.linalg.solve(A, bs)  # [3, n_interior_points]

    # Undo the derivative scaling,  d/dx' → d/dx
    scale = tf.constant([1.0, xscale, yscale], dtype=df.dtype)
    scale = tf.expand_dims(scale, axis=-1)  # for broadcasting
    df = df / scale

    df = tf.reshape(df, (3, *tf.shape(interior_multi_to_linear)))
    return df * zscale


def fit_constant(N: typing.Optional[int],
                 X: typing.Union[np.array, tf.Tensor],
                 Y: typing.Union[np.array, tf.Tensor],
                 Z: typing.Union[np.array, tf.Tensor],
                 *,
                 padding: str,
                 stencil: np.array = None) -> tf.Tensor:
    """[kernel] Like `fit_quadratic`, but fit function values only.

    Each pixel is associated with a local constant model (in an overlapping patch of pixels).

    This can be used as a local least squares denoiser.

    Return value is a rank-3 tensor of shape `[1, ny, nx]` (for API consistency with the other kernels).
    The only channel is the function value.
    """
    if padding.upper() not in ("VALID", "SAME"):
        raise ValueError(f"Invalid padding '{padding}'; valid choices: 'VALID', 'SAME'")
    if stencil is None and N is None:
        raise ValueError("Must specify exactly one of `stencil` or `N`; neither was given.")
    if stencil is not None and N is not None:
        raise ValueError("Cannot specify both a custom `stencil` and `N`. When using a custom `stencil`, please call with `N=None`.")
    if stencil is not None and padding.upper() == "SAME":
        raise ValueError("Cannot use `padding='SAME'` with a custom `stencil`. Please use `padding='VALID'`.")
    if padding.upper() == "SAME":
        assert N is not None
        X = pad_linear_2d(N, X)
        Y = pad_linear_2d(N, Y)
        Z = pad_quadratic_2d(N, Z)
    for name, tensor in (("X", X), ("Y", Y), ("Z", Z)):
        shape = tf.shape(tensor)
        if len(shape) != 2:
            raise ValueError(f"Expected `{name}` to be a rank-2 tensor; got rank {len(shape)}")
        if not (int(shape[0]) >= 2 and int(shape[1]) >= 2):
            raise ValueError(f"Expected `{name}` to be at least of size [2 2]; got {shape}")

    neighbors = stencil if stencil is not None else make_interior_stencil(N)  # [#k, 2]
    min_yoffs = np.min(neighbors[:, 0])
    max_yoffs = np.max(neighbors[:, 0])
    min_xoffs = np.min(neighbors[:, 1])
    max_xoffs = np.max(neighbors[:, 1])

    zscale = tf.reduce_max(tf.abs(Z)).numpy()
    Z = Z / zscale

    # TODO: this is stupid, rewrite the whole kernel for the piecewise constant case.
    def cki(dx, dy):
        one = tf.ones_like(dx)  # for the constant term of the fit
        return np.array([one]).T

    iy, ix = -min_yoffs, -min_xoffs  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    indices = tf.constant(list(zip(iy + neighbors[:, 0], ix + neighbors[:, 1])))
    dx = tf.gather_nd(X, indices) - X[iy, ix]  # [#k]
    dy = tf.gather_nd(Y, indices) - Y[iy, ix]  # [#k]

    # Assemble `A`:
    c = cki(dx, dy)  # [#k, 1]
    A = tf.einsum("ki,kj->ij", c, c)  # A[i,j] = ∑k( c[k,i] * c[k,j] )

    # Form the right-hand side for each point.
    #   b[n,i] = ∑k( f[g[n,k]] * c[k,i] )
    f = tf.reshape(Z, [-1])

    npoints = tf.reduce_prod(tf.shape(X))
    all_multi_to_linear = tf.reshape(tf.range(npoints), tf.shape(X))  # e.g. [0, 1, 2, 3, 4, 5, ...] -> [[0, 1, 2], [3, 4, 5], ...]; C storage order assumed

    ystart = -min_yoffs
    ystop = -max_yoffs if max_yoffs else None
    xstart = -min_xoffs
    xstop = -max_xoffs if max_xoffs else None
    interior_multi_to_linear = all_multi_to_linear[ystart:ystop, xstart:xstop]  # take the valid part

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior pixel
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior pixel

    # Now we can evaluate df, and finally b.
    fgnk = tf.gather(f, gnk)  # [n_interior_points, #k]
    bs = tf.einsum("nk,ki->in", fgnk, c)

    df = tf.linalg.solve(A, bs)  # [1, n_interior_points]

    df = tf.reshape(df, (1, *tf.shape(interior_multi_to_linear)))
    return df * zscale


# --------------------------------------------------------------------------------
# Improved edge handling
#
# Not as good as `prepare`/`solve`, but better than extrapolation with `padding="SAME"`.
#
# We provide two variants, `hifi_differentiate` and `hifier_differentiate`. The quality difference is in the name. :)

def hifi_differentiate(N: int,
                       X: typing.Union[np.array, tf.Tensor],
                       Y: typing.Union[np.array, tf.Tensor],
                       Z: typing.Union[np.array, tf.Tensor],
                       *,
                       kernel: typing.Callable = differentiate):
    """[high-level] Like `differentiate` or `fit_quadratic`, but treat the edges using asymmetric stencils.

    This combines the accuracy advantage of `padding="VALID"` with the range of `padding="SAME"`.
    The results won't be *fully* accurate near the edges, but they do come from the actual data,
    so in general they should be better than a simple quadratic extrapolation (which is what
    `differentiate` and `fit_quadratic` use when `padding="SAME"`).

    `N`: Neighborhood size; must be ≥ 2 to make the equation system solvable also near the corners.
         (There must be at least 6 points in the stencil for `differentiate`, 7 for `fit_quadratic`,
          after the central `(2 * N + 1, 2 * N + 1)` stencil is clipped to the data region.)

    `kernel`: One of the WLSQM differentiation functions, such as `differentiate` or `fit_quadratic`.
    """
    if N < 2:
        raise ValueError(f"`hifi_differentiate` requires N ≥ 2; got {N}.")

    intarray = lambda x: np.array(x, dtype=int)
    ny, nx = tf.shape(Z).numpy()

    interior_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-N, N + 1)])
    interior = kernel(N=None, X=X, Y=Y, Z=Z, padding="VALID", stencil=interior_stencil)
    assert (tf.shape(interior).numpy()[1:] == (ny - 2 * N, nx - 2 * N)).all(), tf.shape(interior)

    top_stencil = intarray([[iy, ix] for iy in range(0, N + 1)
                                     for ix in range(-N, N + 1)])
    top = kernel(N=None, X=X[:(2 * N), :], Y=Y[:(2 * N), :], Z=Z[:(2 * N), :], padding="VALID", stencil=top_stencil)
    assert (tf.shape(top).numpy()[1:] == (N, nx - 2 * N)).all(), tf.shape(top)

    bottom_stencil = intarray([[iy, ix] for iy in range(-N, 1)
                                        for ix in range(-N, N + 1)])
    bottom = kernel(N=None, X=X[-(2 * N):, :], Y=Y[-(2 * N):, :], Z=Z[-(2 * N):, :], padding="VALID", stencil=bottom_stencil)
    assert (tf.shape(bottom).numpy()[1:] == (N, nx - 2 * N)).all(), tf.shape(bottom)

    left_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                      for ix in range(0, N + 1)])
    left = kernel(N=None, X=X[:, :(2 * N)], Y=Y[:, :(2 * N)], Z=Z[:, :(2 * N)], padding="VALID", stencil=left_stencil)
    assert (tf.shape(left).numpy()[1:] == (ny - 2 * N, N)).all(), tf.shape(left)

    right_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                       for ix in range(-N, 1)])
    right = kernel(N=None, X=X[:, -(2 * N):], Y=Y[:, -(2 * N):], Z=Z[:, -(2 * N):], padding="VALID", stencil=right_stencil)
    assert (tf.shape(right).numpy()[1:] == (ny - 2 * N, N)).all(), tf.shape(right)

    ul_stencil = intarray([[iy, ix] for iy in range(0, N + 1)
                                    for ix in range(0, N + 1)])
    ul = kernel(N=None, X=X[:(2 * N), :(2 * N)], Y=Y[:(2 * N), :(2 * N)], Z=Z[:(2 * N), :(2 * N)], padding="VALID", stencil=ul_stencil)
    assert (tf.shape(ul).numpy()[1:] == (N, N)).all(), tf.shape(ul)

    ur_stencil = intarray([[iy, ix] for iy in range(0, N + 1)
                                    for ix in range(-N, 1)])
    ur = kernel(N=None, X=X[:(2 * N), -(2 * N):], Y=Y[:(2 * N), -(2 * N):], Z=Z[:(2 * N), -(2 * N):], padding="VALID", stencil=ur_stencil)
    assert (tf.shape(ur).numpy()[1:] == (N, N)).all(), tf.shape(ur)

    ll_stencil = intarray([[iy, ix] for iy in range(-N, 1)
                                    for ix in range(0, N + 1)])
    ll = kernel(N=None, X=X[-(2 * N):, :(2 * N)], Y=Y[-(2 * N):, :(2 * N)], Z=Z[-(2 * N):, :(2 * N)], padding="VALID", stencil=ll_stencil)
    assert (tf.shape(ll).numpy()[1:] == (N, N)).all(), tf.shape(ll)

    lr_stencil = intarray([[iy, ix] for iy in range(-N, 1)
                                    for ix in range(-N, 1)])
    lr = kernel(N=None, X=X[-(2 * N):, -(2 * N):], Y=Y[-(2 * N):, -(2 * N):], Z=Z[-(2 * N):, -(2 * N):], padding="VALID", stencil=lr_stencil)
    assert (tf.shape(lr).numpy()[1:] == (N, N)).all(), tf.shape(lr)

    # Assemble the output.
    # Data format is [channels, rows, columns].

    # 1) Assemble padded top and bottom edges, with corners.
    fulltop = tf.concat([ul, top, ur], axis=2)  # e.g. [[5, 10, 10], [5, 10, 236], [5, 10, 10]] -> [5, 10, 256]
    fullbottom = tf.concat([ll, bottom, lr], axis=2)

    # 2) Assemble middle part, padding left and right.
    widened = tf.concat([left, interior, right], axis=2)  # e.g. [[5, 236, 10], [5, 236, 236], [5, 236, 10]] -> [5, 236, 256]

    # 3) Assemble the final tensor.
    padded = tf.concat([fulltop, widened, fullbottom], axis=1)  # e.g. [[5, 10, 256], [5, 236, 256], [5, 10, 256]] -> [5, 256, 256]
    return padded


def hifier_differentiate(N: int,
                         X: typing.Union[np.array, tf.Tensor],
                         Y: typing.Union[np.array, tf.Tensor],
                         Z: typing.Union[np.array, tf.Tensor],
                         *,
                         kernel: typing.Callable = differentiate):
    """[high-level] Like `hifi_differentiate`, but with specialized stencils for each row/column near the edge.

    This is slower, but uses all the data near the edges, for a much more accurate estimate especially
    when the data is noisy.

    At the corners, we do a simple approximation based on averaging the two applicable edge handlers at each corner.
    """
    if N < 2:
        raise ValueError(f"`hifi_differentiate` requires N ≥ 2; got {N}.")

    intarray = lambda x: np.array(x, dtype=int)
    ny, nx = tf.shape(Z).numpy()

    # The interior can be handled uniformly, so it costs just one differentiator dispatch (for a large amount of data).
    interior_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-N, N + 1)])
    interior = kernel(N=None, X=X, Y=Y, Z=Z, padding="VALID", stencil=interior_stencil)
    assert (tf.shape(interior).numpy()[1:] == (ny - 2 * N, nx - 2 * N)).all(), tf.shape(interior)

    # Treat the edges.
    #
    # E.g. for N = 2 at the top edge, we can customize the stencil for each row like this:
    #
    #   row 0  row 1  row 2 ... (look as much up as we have data available)
    #   ++x++  +++++  +++++
    #   +++++  ++x++  +++++
    #   +++++  +++++  ++x++
    #          +++++  +++++
    #                 +++++
    #
    # Compare `hifi_differentiate`, which uses the "row 0" type of stencil for all rows near the upper edge.
    # Our strategy costs N differentiator dispatches per edge.
    #
    def treat_top(ix_start=-N, ix_stop=N + 1, datax_start=0, datax_stop=None):
        rows = []
        for row in range(N):
            top_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                             for ix in range(ix_start, ix_stop)])
            this_row = kernel(N=None,
                              X=X[:(N + row + 1), datax_start:datax_stop],
                              Y=Y[:(N + row + 1), datax_start:datax_stop],
                              Z=Z[:(N + row + 1), datax_start:datax_stop],
                              padding="VALID", stencil=top_stencil)
            assert tf.shape(this_row).numpy()[1] == 1, tf.shape(this_row)  # one row
            rows.append(this_row)
        return tf.concat(rows, axis=1)
    top = treat_top()
    assert (tf.shape(top).numpy()[1:] == (N, nx - 2 * N)).all(), tf.shape(top)

    def treat_bottom(ix_start=-N, ix_stop=N + 1, datax_start=0, datax_stop=None):
        rows = []
        for row in range(-N, 0):
            bottom_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                                for ix in range(ix_start, ix_stop)])
            this_row = kernel(N=None,
                              X=X[-(N - row):, datax_start:datax_stop],
                              Y=Y[-(N - row):, datax_start:datax_stop],
                              Z=Z[-(N - row):, datax_start:datax_stop],
                              padding="VALID", stencil=bottom_stencil)
            assert tf.shape(this_row).numpy()[1] == 1, tf.shape(this_row)  # one row
            rows.append(this_row)
        return tf.concat(rows, axis=1)
    bottom = treat_bottom()
    assert (tf.shape(bottom).numpy()[1:] == (N, nx - 2 * N)).all(), tf.shape(bottom)

    #  col0  col1  col2  ... (look as much to the left as we have data available)
    #   +++  ++++  +++++
    #   +++  ++++  +++++
    #   x++  +x++  ++x++
    #   +++  ++++  +++++
    #   +++  ++++  +++++
    def treat_left(iy_start=-N, iy_stop=N + 1, datay_start=0, datay_stop=None):
        cols = []
        for col in range(N):
            left_stencil = intarray([[iy, ix] for iy in range(iy_start, iy_stop)
                                              for ix in range(-col, N + 1)])
            this_col = kernel(N=None,
                              X=X[datay_start:datay_stop, :(N + col + 1)],
                              Y=Y[datay_start:datay_stop, :(N + col + 1)],
                              Z=Z[datay_start:datay_stop, :(N + col + 1)],
                              padding="VALID", stencil=left_stencil)
            assert tf.shape(this_col).numpy()[2] == 1, tf.shape(this_col)  # one column
            cols.append(this_col)
        return tf.concat(cols, axis=2)
    left = treat_left()
    assert (tf.shape(left).numpy()[1:] == (ny - 2 * N, N)).all(), tf.shape(left)

    def treat_right(iy_start=-N, iy_stop=N + 1, datay_start=0, datay_stop=None):
        cols = []
        for col in range(-N, 0):
            right_stencil = intarray([[iy, ix] for iy in range(iy_start, iy_stop)
                                               for ix in range(-N, -col)])
            this_col = kernel(N=None,
                              X=X[datay_start:datay_stop, -(N - col):],
                              Y=Y[datay_start:datay_stop, -(N - col):],
                              Z=Z[datay_start:datay_stop, -(N - col):],
                              padding="VALID", stencil=right_stencil)
            assert tf.shape(this_col).numpy()[2] == 1, tf.shape(this_col)  # one column
            cols.append(this_col)
        return tf.concat(cols, axis=2)
    right = treat_right()
    assert (tf.shape(right).numpy()[1:] == (ny - 2 * N, N)).all(), tf.shape(right)

    # The most accurate solution in the corners would be to use a per-pixel customized stencil,
    # at a cost of N² differentiator dispatches per corner (one for each pixel in the corner region).
    #
    # Let's try a cheaper solution, which also allows us to reuse the edge handlers: use both applicable
    # variants of edge treatment, and average them, at a cost of 2*N dispatches per corner.
    #
    # That is, e.g. at the upper left, with N = 2, average the result from these stencils (top edge treatment):
    #
    #   row 0  row 1  row 2 ... (to accommodate corner, look only to the right)
    #     x++    +++    +++
    #     +++    x++    +++
    #     +++    +++    x++
    #            +++    +++
    #                   +++
    #
    # with the result from these (left edge treatment):
    #
    #   col 0 col 1  col 2 ... (to accommodate corner, look only down)
    #     x++  +x++  ++x++
    #     +++  ++++  +++++
    #     +++  ++++  +++++
    #
    # Feeding in a data region of size [2 * N, 2 * N] then yields a result of size [N, N].
    #
    # The quality is worst at the corner pixel, where both stencils coincide, and are only of size [N + 1, N + 1].
    # There's not much that can be done about that, except use a larger neighborhood and risk numerical instability.
    # Even the optimal N² method would have this limitation.
    ul1 = treat_top(ix_start=0, datax_stop=2 * N)
    ul2 = treat_left(iy_start=0, datay_stop=2 * N)
    ul = (ul1 + ul2) / 2
    assert (tf.shape(ul).numpy()[1:] == (N, N)).all(), tf.shape(ul)

    ur1 = treat_top(ix_stop=1, datax_start=-2 * N)
    ur2 = treat_right(iy_start=0, datay_stop=2 * N)
    ur = (ur1 + ur2) / 2
    assert (tf.shape(ur).numpy()[1:] == (N, N)).all(), tf.shape(ur)

    ll1 = treat_bottom(ix_start=0, datax_stop=2 * N)
    ll2 = treat_left(iy_stop=1, datay_start=-2 * N)
    ll = (ll1 + ll2) / 2
    assert (tf.shape(ll).numpy()[1:] == (N, N)).all(), tf.shape(ll)

    lr1 = treat_bottom(ix_stop=1, datax_start=-2 * N)
    lr2 = treat_right(iy_stop=1, datay_start=-2 * N)
    lr = (lr1 + lr2) / 2
    assert (tf.shape(lr).numpy()[1:] == (N, N)).all(), tf.shape(lr)

    # Assemble the output.
    # Data format is [channels, rows, columns].

    # 1) Assemble padded top and bottom edges, with corners.
    fulltop = tf.concat([ul, top, ur], axis=2)  # e.g. [[5, 10, 10], [5, 10, 236], [5, 10, 10]] -> [5, 10, 256]
    fullbottom = tf.concat([ll, bottom, lr], axis=2)

    # 2) Assemble middle part, padding left and right.
    widened = tf.concat([left, interior, right], axis=2)  # e.g. [[5, 236, 10], [5, 236, 236], [5, 236, 10]] -> [5, 236, 256]

    # 3) Assemble the final tensor.
    padded = tf.concat([fulltop, widened, fullbottom], axis=1)  # e.g. [[5, 10, 256], [5, 236, 256], [5, 10, 256]] -> [5, 256, 256]
    return padded
