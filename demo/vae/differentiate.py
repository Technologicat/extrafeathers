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

import gc
import math
import typing

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
    """Get the size of `tf.Tensor` or `tf.RaggedTensor`, in bytes.

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

def prepare(N: float,
            X: typing.Union[np.array, tf.Tensor],
            Y: typing.Union[np.array, tf.Tensor],
            Z: typing.Union[np.array, tf.Tensor],
            *,
            p: typing.Union[float, str] = 2.0,
            dtype: tf.DType = tf.float32,
            format: str = "A",
            low_vram: bool = False,
            print_memory_statistics: bool = False):
    """Prepare for differentiation on a meshgrid.

    This is the hifiest algorithm provided in this module. For what to do after `prepare`, see `solve`.

    This function precomputes the surrogate fitting coefficient tensor `c`, and the pixelwise `A` matrices.
    Some of the preparation is performed on the CPU, the most intensive computations on the GPU.

    NOTE: This takes a lot of VRAM. A 6 GB GPU is able to do 256×256, but not much more.

    `N`: Neighborhood size parameter (in grid spacings).

         The edges and corners of the input image are handled by clipping the stencil to the data region
         (to use as much data for each pixel as exists within distance `N` on each axis).

         Non-integer values can be useful with float values of `p` (see below). This affects the shape
         of the outer edge of the neighborhood. (E.g., with `p = 2.0`, integer `N` has a 1-pixel protrusion
         along the cardinal directions, although otherwise the stencil is a good approximation of a round shape.
         Using e.g. `N=10.9` instead of `N=11` can avoid this.)

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
             The returned `tensors` (see below) are `(A, c, scale, neighbors)`.

        "LUp": Format expected by `solve_lu` (which can run also at float16).
             The returned `tensors` (see below) are `(LU, p, c, scale, neighbors)`.

        The returned tensors can be passed to `solve` or `solve_lu` (depending on `format`);
        these solvers run completely on the GPU.

    `low_vram`: If `True`, store the coefficient tensor `c` in half precision (float16).
                This makes `A` slightly less precise, but not much, while it halves the VRAM
                requirements for the largest tensor in the preparation process.

    `print_memory_statistics`: If `True`, print on stdout how much memory (usually VRAM) the
                               various tensors used during the preparation process use.

                               This of course only succeeds if there is enough memory to
                               actually allocate those tensors. The idea is to help give
                               a rough intuition of the scaling behavior.

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
        return np.array(x, dtype=int)  # TODO: `np.int32`?

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
    npoints = tf.reduce_prod(shape)
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
    # We create a per-datapoint indirection tensor, mapping datapoint linear index to the index of the appropriate unique stencil.
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
    indirect = np.zeros([npoints], dtype=int) - 1  # -1 = uninitialized, to catch bugs
    def register_stencil(stencil, for_points):  # set up indirection
        stencils.append(stencil)
        stencil_id = len(stencils) - 1
        indirect[for_points] = stencil_id
        return stencil_id

    # Interior - one stencil for all pixels; this case handles almost all of the image.
    interior_multi_to_linear = all_multi_to_linear[N:-N, N:-N]  # take the interior part of the meshgrid
    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior data point (C storage order)
    interior_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-N, N + 1)
                                          if belongs_to_neighborhood(iy, ix)])  # multi-index offsets
    orig_interior_stencil = interior_stencil  # for returning to caller
    interior_stencil = multi_to_linear(interior_stencil, shape=shape)  # corresponding linear index offsets
    register_stencil(interior_stencil, interior_idx)

    # Top edge - one stencil per row (N of them, so typically 8).
    for row in range(N):
        top_multi_to_linear = all_multi_to_linear[row, N:-N]
        top_idx = tf.reshape(top_multi_to_linear, [-1])
        top_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                         for ix in range(-N, N + 1)
                                         if belongs_to_neighborhood(iy, ix)])
        top_stencil = multi_to_linear(top_stencil, shape=shape)
        register_stencil(top_stencil, top_idx)  # each row near the top gets its own stencil

    # Bottom edge - one stencil per row.
    for row in range(-N, 0):
        bottom_multi_to_linear = all_multi_to_linear[row, N:-N]
        bottom_idx = tf.reshape(bottom_multi_to_linear, [-1])
        bottom_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                            for ix in range(-N, N + 1)
                                            if belongs_to_neighborhood(iy, ix)])
        bottom_stencil = multi_to_linear(bottom_stencil, shape=shape)
        register_stencil(bottom_stencil, bottom_idx)

    # Left edge - one stencil per column (N of them, so typically 8).
    for col in range(N):
        left_multi_to_linear = all_multi_to_linear[N:-N, col]
        left_idx = tf.reshape(left_multi_to_linear, [-1])
        left_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-col, N + 1)
                                          if belongs_to_neighborhood(iy, ix)])
        left_stencil = multi_to_linear(left_stencil, shape=shape)
        register_stencil(left_stencil, left_idx)

    # Right edge - one stencil per column.
    for col in range(-N, 0):
        right_multi_to_linear = all_multi_to_linear[N:-N, col]
        right_idx = tf.reshape(right_multi_to_linear, [-1])
        right_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                           for ix in range(-N, -col)
                                           if belongs_to_neighborhood(iy, ix)])
        right_stencil = multi_to_linear(right_stencil, shape=shape)
        register_stencil(right_stencil, right_idx)

    # Upper left corner - one stencil per pixel (N² of them, so typically 64).
    for row in range(N):
        for col in range(N):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])  # just one pixel, but for uniform data format, use a rank-1 tensor
            this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                              for ix in range(-col, N + 1)
                                              if belongs_to_neighborhood(iy, ix)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Upper right corner - one stencil per pixel.
    for row in range(N):
        for col in range(-N, 0):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                              for ix in range(-N, -col)
                                              if belongs_to_neighborhood(iy, ix)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Lower left corner - one stencil per pixel.
    for row in range(-N, 0):
        for col in range(N):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                              for ix in range(-col, N + 1)
                                              if belongs_to_neighborhood(iy, ix)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Lower right corner - one stencil per pixel.
    for row in range(-N, 0):
        for col in range(-N, 0):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                              for ix in range(-N, -col)
                                              if belongs_to_neighborhood(iy, ix)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    assert len(stencils) == 1 + 4 * N + 4 * N**2  # interior, edges, corners
    assert not (indirect == -1).any()  # every pixel of the input image should now have a stencil associated with it

    # For meshgrid use, we can store stencils as lists of linear index offsets (int32), in a ragged tensor.
    # Ragged, because points near edges or corners have a clipped stencil with fewer neighbors.
    indirect = tf.constant(indirect, dtype=tf.int32)
    stencils = tf.ragged.constant(stencils, dtype=tf.int32)

    if print_memory_statistics:
        print(f"indirect: {sizeof_tensor(indirect)}, {indirect.dtype}")
        print(f"stencils: {sizeof_tensor(stencils)}, {stencils.dtype}")

    # Build the distance matrices.
    #
    # First apply derivative scaling for numerical stability: x' := x / xscale  ⇒  d/dx → (1 / xscale) d/dx'.
    #
    # We choose xscale to make magnitudes near 1. Similarly for yscale. We base the scaling on the grid spacing in raw coordinate space,
    # defining the furthest distance in the stencil (along each coordinate axis) in the scaled space as 1.
    #
    # TODO: We assume a uniform grid spacing for now. We could relax this assumption, since only the choice of `xscale`/`yscale` depend on it.
    #
    # We cast to `float`, so this works also in the case where we get a scalar tensor instead of a bare scalar.
    xscale = float(X[0, 1] - X[0, 0]) * N
    yscale = float(Y[1, 0] - Y[0, 0]) * N

    # --------------------------------------------------------------------------------
    # General, topology-agnostic assembly

    def cnki(dx: tf.Tensor, dy: tf.Tensor) -> tf.Tensor:
        """Compute the quadratic surrogate fitting coefficient tensor `c[n, k, i]`.

        NOTE: The `c` tensor may take gigabytes of VRAM, depending on input image resolution.

        Input: rank-2 tensors:

          `dx`: signed `x` distance
          `dy`: signed `y` distance

        where:

          dx[n, k] = signed x distance from point n to point k

        and similarly for `dy`. The indices are:

          `n`: linear index of data point
          `k`: linear index (not offset!) of neighbor point

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
    X = tf.cast(X, dtype)
    Y = tf.cast(Y, dtype)
    # We'll be using linear indexing.
    X = tf.reshape(X, [-1])
    Y = tf.reshape(Y, [-1])
    if print_memory_statistics:
        print(f"X: {sizeof_tensor(X)}, {X.dtype}")
        print(f"Y: {sizeof_tensor(Y)}, {Y.dtype}")
    # `neighbors`: linear indices (not offsets!) of neighbors (in stencil) for each pixel; resolution² * (2 * N + 1)² * 4 bytes.
    #              The first term is the base linear index of each data point; the second is the linear index offset of each of its neighbors.
    neighbors = tf.expand_dims(tf.range(npoints), axis=-1) + tf.gather(stencils, indirect)
    if print_memory_statistics:
        print(f"neighbors: {sizeof_tensor(neighbors)}, {neighbors.dtype}")  # Spoiler: this tensor is moderately large.

    # `dx[n, k]`: signed x distance of neighbor `k` from data point `n`. Similarly for `dy[n, k]`.
    # TODO: Low VRAM mode: This is the straw that breaks the camel's back for very large neighborhoods (N = 20.5).
    dx = tf.gather(X, neighbors) - tf.expand_dims(X, axis=-1)  # `expand_dims` explicitly, to broadcast on the correct axis
    dy = tf.gather(Y, neighbors) - tf.expand_dims(Y, axis=-1)
    if print_memory_statistics:
        print(f"dx: {sizeof_tensor(dx)}, {dx.dtype}")
        print(f"dy: {sizeof_tensor(dy)}, {dy.dtype}")

    # Finally, the surrogate fitting coefficient tensor is the following `c`.
    #
    # Currently, `c` takes a lot of VRAM, even at float16. It would be possible to save VRAM by a similar indirection strategy
    # as is already used for stencils. Specifically, in a uniform meshgrid, each unique stencil corresponds to a unique `c[n, :, :]`.
    # So by indirecting when assembling the equations later, we would strictly need to store only one `c[n, :, :]` per stencil,
    # not per pixel as we do currently.
    #
    # However, I'm tempted to keep this part as-is, because the current approach is general. It would work also with arbitrary topologies,
    # where every "pixel" (really a mesh node; or a point of a generic, topology-free point cloud) has a unique stencil.
    #
    # Note also that we can already assemble reasonable sizes (N=13, p=2.0) with 6 GB.
    #
    c = cnki(dx, dy)
    if print_memory_statistics:
        print(f"c: {sizeof_tensor(c)}, {c.dtype}")  # Spoiler: this tensor is huge.

    # # DEBUG: If the x and y scalings work, the range of values in `c` should be approximately [0, 1].
    # absc = tf.abs(c)
    # print(f"c[n, k, i] ∈ [{tf.reduce_min(absc):0.6g}, {tf.reduce_max(absc):0.6g}]")

    # The surrogate fitting tensor is:
    #
    # A[n,i,j] = ∑k( c[n,k,i] * c[n,k,j] )
    #
    # The `A` matrices can be preassembled. They must be stored per-pixel, but the size is only 6×6, so at float32,
    # we need 36 * 4 bytes * resolution² = 144 * resolution², which is only 38 MB at 512×512, and at 1024×1024, only 151 MB.
    #
    # `einsum` doesn't support `RaggedTensor`. What we want to do:
    # # A = tf.einsum("nki,nkj->nij", c, c)
    # Doing it manually:
    rows = []
    for i in range(6):
        row = []
        for j in range(6):
            # Always use float32 to compute the elements of `A` for optimal accuracy, even if our storage `dtype` happens to be float16.
            ci = tf.cast(c[:, :, i], tf.float32)  # -> [#n, #k], where #k is ragged (number of neighbors in stencil for pixel `n`)
            cj = tf.cast(c[:, :, j], tf.float32)  # -> [#n, #k]
            # TODO: Low VRAM: here is another straw that breaks the camel's back (N=17.5). We should batch the assembly of `A`, like we already do for `b`.
            Aij = tf.reduce_sum(ci * cj, axis=1)  # [#n, #k] -> [#n]
            row.append(tf.cast(Aij, dtype))
        row = tf.stack(row, axis=1)  # -> [#n, #cols]
        rows.append(row)
    A = tf.stack(rows, axis=1)  # [[#n, #cols], [#n, #cols], ...] -> [#n, #rows, #cols]
    if print_memory_statistics:
        print(f"A: {sizeof_tensor(A)}, {A.dtype}")

    # # DEBUG: If the x and y scalings work, the range of values in `A` should be approximately [0, (2 * N + 1)²].
    # # The upper bound comes from the maximal number of points in the stencil, and is reached when gathering this many "ones" in the constant term.
    # absA = tf.abs(A)
    # print(f"A[n, i, j] ∈ [{tf.reduce_min(absA):0.6g}, {tf.reduce_max(absA):0.6g}]")

    # TODO: DEBUG: sanity check that each `A[n, :, :]` is symmetric.

    # Scaling factors to undo the derivative scaling,  d/dx' → d/dx.  `solve` needs this to postprocess its results.
    scale = tf.constant([1.0, xscale, yscale, xscale**2, xscale * yscale, yscale**2], dtype=dtype)
    # scale = tf.expand_dims(scale, axis=-1)  # for broadcasting; solution shape from `tf.linalg.solve` is [6, npoints]
    if print_memory_statistics:
        print(f"scale: {sizeof_tensor(scale)}, {scale.dtype}")

    if format == "LUp":
        LU, p = tf.linalg.lu(tf.cast(A, tf.float32))
        LU = tf.cast(LU, dtype)
        if print_memory_statistics:
            print(f"LU: {sizeof_tensor(LU)}, {LU.dtype}")
            print(f"p: {sizeof_tensor(p)}, {p.dtype}")

    del dx
    del dy
    gc.collect()  # attempt to clean up dangling tensors

    if format == "A":
        return (A, c, scale, neighbors), orig_interior_stencil
    return (LU, p, c, scale, neighbors), orig_interior_stencil


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
# preserved as `solve_lu`.

@tf.function
def _assemble_b(c: tf.Tensor,
                neighbors: tf.RaggedTensor,
                z: tf.Tensor,
                low_vram: bool,
                low_vram_batches: int) -> tf.Tensor:
    """[internal helper] Assemble the load vector.

    For the parameters, see `solve`.

    Returns a `tf.Tensor` of shape [npoints, 6].
    """
    # b[n,i] = ∑k( z[neighbors[n,k]] * c[n,k,i] )
    if not low_vram:
        # The `RaggedSplitsToSegmentIds`, used internally by TF inside the `reduce_sum` costs a lot of VRAM, since it indexes using `int64`,
        # and we're handling a lot of tensor elements here. E.g. at 256×256, #n = 65536, and with N=13, p=2.0, #k ~ 500, we have about 30M points;
        # so with an int64 for each, we're talking ~300 MB just for the indexing. When the chosen settings are already pushing against the VRAM limit,
        # this is the straw that breaks the camel's back.
        #
        # When enough VRAM is available, this is the simplest way to do the assembly:
        rows = []
        zgnk = tf.gather(z, neighbors, name="gather_neighbors")  # -> [#n, #k], ragged in k
        for i in range(6):
            ci = tf.cast(c[:, :, i], tf.float32, name="cast_to_float32")  # -> [#n, #k]
            bi = tf.reduce_sum(zgnk * ci, axis=1, name="assemble_bi")  # [#n, #k] -> [#n]
            rows.append(tf.cast(bi, z.dtype, name="cast_to_dtype"))
    else:
        # In low VRAM mode, we assemble `b` in batches, splitting over the meshgrid points.
        npoints = int(tf.shape(z)[0])  # NOTE: reshaped, linearly indexed `z`
        batches = [[j * npoints // low_vram_batches, (j + 1) * npoints // low_vram_batches] for j in range(low_vram_batches)]
        batches[-1][-1] = None  # Set the final `stop` value to include all remaining tensor elements in the final batch.
        rows_by_batch = []
        for start, stop in batches:
            rows = []
            zgnk_split = tf.gather(z, neighbors[start:stop], name="gather_neighbors")  # [#n] -> [#split, #k], ragged in k
            for i in range(6):
                ci_split = tf.cast(c[start:stop, :, i], tf.float32, name="cast_to_float32")  # -> [#split, #k]
                bi_split = tf.reduce_sum(zgnk_split * ci_split, axis=1, name="assemble_bi")  # [#split, #k] -> [#split]
                rows.append(tf.cast(bi_split, z.dtype, name="cast_to_dtype"))
            rows_by_batch.append(rows)
        # Unsplit:
        #   [[batch0_row0, batch0_row1, ..., batch0_row5], [batch1_row0, batch1_row1, ..., batch1_row5], ...] -> [row0, row1, ..., row5]
        # `b` itself doesn't take much VRAM (less than 2 MB in the above scenario), so we don't have to worry about VRAM usage here.
        rows = []
        for rs in zip(*rows_by_batch):
            rows.append(tf.concat(rs, axis=0))  # [#split] -> [#n]

    b = tf.stack(rows, axis=1, name="stack_b")  # -> [#n, #rows]
    return b

@tf.function
def solve(a: tf.Tensor,
          c: tf.Tensor,
          scale: tf.Tensor,
          neighbors: tf.RaggedTensor,
          z: tf.Tensor,
          low_vram: bool = True,
          low_vram_batches: int = 4) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    This uses `tf.linalg.solve`, and is the recommended algorithm.

    `a`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="A"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batches`: If `low_vram=True`, this is the number of batches for assembling the load vector.
                        Increase this to trade off speed for lower VRAM usage.

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

    b = _assemble_b(c, neighbors, z, low_vram, low_vram_batches)  # -> [#n, #rows]

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
             neighbors: tf.RaggedTensor,
             z: tf.Tensor,
             low_vram: bool = True,
             low_vram_batches: int = 4) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    This uses a TensorFlow's LU solver kernel.

    `lu`, `p`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="LUp"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batches`: If `low_vram=True`, this is the number of batches for assembling the load vector.
                        Increase this to trade off speed for lower VRAM usage.

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

    b = _assemble_b(c, neighbors, z, low_vram, low_vram_batches)  # -> [#n, #rows]

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
                    neighbors: tf.RaggedTensor,
                    z: tf.Tensor,
                    low_vram: bool = True,
                    low_vram_batches: int = 4) -> tf.Tensor:
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    This uses a custom LU solver kernel, which can run also at float16.

    `lu`, `p`, `c`, `scale`, `neighbors`: Outputs from `prepare` with `format="LUp"`, which see.
    `z`: function value data in 2D meshgrid format.

         For computations, `z` is automatically cast into the proper dtype.

    `low_vram`: If `True`, attempt to save VRAM by splitting the load vector assembly process
                into batches over the meshgrid points.

                This will slow down the computation (especially at first run when TF compiles the graph),
                but allows using larger neighborhood sizes with the same VRAM.

    `low_vram_batches`: If `low_vram=True`, this is the number of batches for assembling the load vector.
                        Increase this to trade off speed for lower VRAM usage.

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
    return _solve_lu_custom_kernel(lu, p, c, scale, neighbors, z, low_vram, low_vram_batches, x)

@tf.function
def _solve_lu_custom_kernel(lu: tf.Tensor,
                            p: tf.Tensor,
                            c: tf.Tensor,
                            scale: tf.Tensor,
                            neighbors: tf.RaggedTensor,
                            z: tf.Tensor,
                            low_vram: bool,
                            low_vram_batches: int,
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

    b = _assemble_b(c, neighbors, z, low_vram, low_vram_batches)  # -> [#n, #rows]

    # We must call the kernel directly, because we are inside @tf.function;
    # the API wrapper would try to allocate its own `x`, which we can't do here.
    lusolve(lu, p, b, x)  # our custom kernel, writes solution into `x`
    sol = x

    sol = sol / scale  # return derivatives from scaled x, y (as set up by `prepare`) to raw x, y
    sol = zmax * sol  # return from scaled z to raw z

    sol = tf.transpose(sol, [1, 0])  # -> [#rows, #n]
    return tf.reshape(sol, (6, int(shape[0]), int(shape[1])))  # -> [#rows, ny, nx]


# --------------------------------------------------------------------------------
# Basic API. Slower, not as accurate in corners, needs less VRAM.
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

    Each data point is associated with a local quadratic model.

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
        with 6 or more data points (xk, yk, fk), we can write a linear equation system that yields approximate
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
            - `float`, for a single pair of data points
            - rank-1 `np.array`, for a batch of data point pairs

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
    #   `n`: datapoint in batch,
    #   `k`: neighbor,
    #   `i`: row of MLS equation system "A x = b"
    #   `j`: column of MLS equation system "A x = b"
    # and let `f[n] = f(x[n], y[n])`.
    #
    # Then, in the general case, the MLS equation systems for the batch are given by:
    #   A[n,i,j] = ∑k( c[n,k,i] * c[n,k,j] )
    #   b[n,i] = ∑k( (f[g[n,k]] - f[n]) * c[n,k,i] )
    #
    # where `g[n,k]` is the (global) data point index, of neighbor `k` of data point `n`.
    #
    # On a uniform grid, c[n1,k,i] = c[n2,k,i] =: c[k,i] for any n1, n2, so this simplifies to:
    #   A[i,j] = ∑k( c[k,i] * c[k,j] )
    #   b[n,i] = ∑k( (f[g[n,k]] - f[n]) * c[k,i] )
    #
    # In practice we still have to chop the edges, which modifies the data point indexing slightly.
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

    # Form the right-hand side for each point. This is the only part that depends on the data values f.
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

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior data point
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior data point

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

    # The solution of the linear systems (one per data point) yields the jacobian and hessian of the surrogate.
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

    Each data point is associated with a local quadratic model.

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
    # where `g[n,k]` is the (global) data point index, of neighbor `k` of data point `n`.
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

    # Form the right-hand side for each point. This is the only part that depends on the data values f.
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

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior data point
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior data point

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

    Each data point is associated with a local linear model.

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

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior data point
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior data point

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

    Each data point is associated with a local constant model.

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

    interior_idx = tf.reshape(interior_multi_to_linear, [-1])  # [n_interior_points], linear index of each interior data point
    # linear index, C storage order: i = iy * size_x + ix
    offset_idx = neighbors[:, 0] * tf.shape(X)[1] + neighbors[:, 1]  # [#k], linear index *offset* for each neighbor in the neighborhood

    # Compute index sets for df. Use broadcasting to create an "outer sum" [n,1] + [1,k] -> [n,k].
    n = tf.expand_dims(interior_idx, axis=1)  # [n_interior_points, 1]
    offset_idx = tf.expand_dims(offset_idx, axis=0)  # [1, #k]
    gnk = n + offset_idx  # [n_interior_points, #k], linear index of each neighbor of each interior data point

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
