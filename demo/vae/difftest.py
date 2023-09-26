#!/usr/bin/env python
"""Use the weighted least squares meshfree method to fit a local quadratic surrogate to compute derivatives.

This method produces approximations for both the jacobian and the hessian in one go, by solving local 5×5 equation systems.

This is basically a technique demo; this can be GPU-accelerated with TF, so we can use it to evaluate the
spatial derivatives in the PDE residual in a physically informed loss for up to 2nd order PDEs.

A potential issue is that the VAE output might contain some noise, so to be safe, we need a method that can handle noisy input.

We need only the very basics here. A complete Cython implementation, and documentation:
    https://github.com/Technologicat/python-wlsqm
"""

import numpy as np
import sympy as sy
import tensorflow as tf

import matplotlib.pyplot as plt


# TODO: move to `extrafeathers.plotmagic`; or create a new small module for `pause` and this, since the rest is FEniCS-specific and these are general.
def link_3d_subplot_cameras(fig, axes):
    """Link view angles and zooms of several 3d subplots in the same Matplotlib figure.

    `fig`: a Matplotlib figure object
    `axes`: a list of Matplotlib `Axes` objects in figure `fig`

    Return value is a handle to a Matplotlib `motion_notify_event`;
    you can disconnect this event later if you want to unlink the cameras.

    Example::

        fig = plt.figure(1)
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax2 = fig.add_subplot(2, 3, 1, projection="3d")
        link_3d_subplot_cameras(fig, [ax1, ax2])

    Adapted from recipe at:
        https://github.com/matplotlib/matplotlib/issues/11181
    """
    def on_move(event):
        sender = [ax for ax in axes if event.inaxes == ax]
        if not sender:
            return
        assert len(sender) == 1
        sender = sender[0]
        others = [ax for ax in axes if ax is not sender]
        if sender.button_pressed in sender._rotate_btn:
            for ax in others:
                ax.view_init(elev=sender.elev, azim=sender.azim)
        elif sender.button_pressed in sender._zoom_btn:
            for ax in others:
                ax.set_xlim3d(sender.get_xlim3d())
                ax.set_ylim3d(sender.get_ylim3d())
                ax.set_zlim3d(sender.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
    return fig.canvas.mpl_connect("motion_notify_event", on_move)


def make_stencil(N: int) -> np.array:
    """Return an array of integer offset pairs for a square-shaped stencil of N×N points.

    `N`: neighborhood size parameter (how many grid spacings on each axis)

    Return value is a rank-2 np-array of shape [n_neighbors, 2].
    """
    neighbors = [[iy, ix] for iy in range(-N, N + 1)
                          for ix in range(-N, N + 1)
                          if not (iy == 0 and ix == 0)]
    neighbors = np.array(neighbors, dtype=int)
    return neighbors


# The edges are nonsense with padding="SAME", so we use "VALID", and chop off the edges of X and Y correspondingly.
def chop_edges(N: int, X, Y):
    return X[N:-N, N:-N], Y[N:-N, N:-N]


# --------------------------------------------------------------------------------
# Tensor padding by extrapolation

@tf.function
def _assemble_padded(*, interior: tf.Tensor,
                     top: tf.Tensor, bottom: tf.Tensor, left: tf.Tensor, right: tf.Tensor,
                     top_left: float, top_right: float, bottom_left: float, bottom_right: float):
    """Assemble a padded rank-2 tensor from parts.

    `interior`: rank-2 tensor, [nrows, ncols]; original tensor to which the padding is to be added
    `top`, `bottom`: rank-1 tensors, [ncols]; padding edge values
    `left`, `right`: rank-1 tensors, [nrows]; padding edge values
    `top_left`, `top_right`, `bottom_left`, `bottom_right`: scalars; padding corner values

    Return value is a rank-2 tensor, populated as follows::

        TL T T T T TR
         L i i i i R
         L i i i i R
         L i i i i R
        BL B B B B BR
    """
    # 1) Assemble padded top and bottom edges, with corners.
    ult = tf.expand_dims(top_left, axis=0)  # [] -> [1]
    urt = tf.expand_dims(top_right, axis=0)
    fulltop = tf.concat([ult, top, urt], axis=0)  # e.g. [256] -> [258]
    fulltop = tf.expand_dims(fulltop, axis=0)  # [258] -> [1, 258]
    llt = tf.expand_dims(bottom_left, axis=0)
    lrt = tf.expand_dims(bottom_right, axis=0)
    fullbottom = tf.concat([llt, bottom, lrt], axis=0)
    fullbottom = tf.expand_dims(fullbottom, axis=0)

    # 2) Assemble middle part, padding left and right.
    left = tf.expand_dims(left, axis=-1)  # [256] -> [256, 1]
    right = tf.expand_dims(right, axis=-1)
    widened = tf.concat([left, interior, right], axis=1)  # [256, 256] -> [256, 258]

    # 3) Assemble the final tensor.
    padded = tf.concat([fulltop, widened, fullbottom], axis=0)  # -> [258, 258]
    return padded


@tf.function(reduce_retracing=True)
def pad_constant_one(f):
    """Pad 2D tensor by one grid unit, by copying the nearest value from the edges.

    `f`: data in meshgrid format.

    This is really "nearest", but "constant" in the sense of the other paddings defined here.
    In other words, we estimate the function value as locally constant.

    The corner paddings are generated by applying the same idea diagonally.
    """
    top = f[0, :]
    bottom = f[-1, :]
    left = f[:, 0]
    right = f[:, -1]

    tl = f[0, 0]
    tr = f[0, -1]
    bl = f[-1, 0]
    br = f[-1, -1]

    return _assemble_padded(interior=f, top=top, bottom=bottom, left=left, right=right,
                            top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)

@tf.function(reduce_retracing=True)
def pad_linear_one(f):
    """Pad 2D tensor by one grid unit, by linear extrapolation.

    `f`: data in meshgrid format.

    For example, if the layout at the end of a row is::

      f0 f1 | f2

    where f2 is to be created by extrapolation, then::

      f2 = f1 + Δf
         = f1 + [f1 - f0]
         = 2 f1 - f0

    where we estimate the first difference Δf as locally constant; or in other words,
    we estimate the function value itself as locally linear::

      Δf := f1 - f0 = f2 - f1

    The corner paddings are generated by applying the same idea diagonally.
    """
    top = 2 * f[0, :] - f[1, :]
    bottom = 2 * f[-1, :] - f[-2, :]
    left = 2 * f[:, 0] - f[:, 1]
    right = 2 * f[:, -1] - f[:, -2]

    tl = 2 * f[0, 0] - f[1, 1]
    tr = 2 * f[0, -1] - f[1, -2]
    bl = 2 * f[-1, 0] - f[-2, 1]
    br = 2 * f[-1, -1] - f[-2, -2]

    return _assemble_padded(interior=f, top=top, bottom=bottom, left=left, right=right,
                            top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)

@tf.function(reduce_retracing=True)
def pad_quadratic_one(f):
    """Pad 2D tensor by one grid unit, by quadratic extrapolation.

    `f`: data in meshgrid format.

    For example, if the layout at the end of a row is::

      f0 f1 f2 | f3

    where f3 is to be created by extrapolation, then::

      f3 = f2 + Δf_23

    where we estimate the first difference Δf as locally linear; or in other words,
    we estimate the function value itself as locally quadratic::

      Δf_23 = Δf_12 + [Δf_12 - Δf_01]
            = 2 Δf_12 - Δf_01
            ≡ 2 [f2 - f1] - [f1 - f0]
            = 2 f2 - 3 f1 + f0

    so finally::

      f3 = f2 + [2 f2 - 3 f1 + f0]
         = 3 f2 - 3 f1 + f0

    The corner paddings are generated by applying the same idea diagonally.
    """
    top = 3 * f[0, :] - 3 * f[1, :] + f[2, :]
    bottom = 3 * f[-1, :] - 3 * f[-2, :] + f[-3, :]
    left = 3 * f[:, 0] - 3 * f[:, 1] + f[:, 2]
    right = 3 * f[:, -1] - 3 * f[:, -2] + f[:, -3]

    tl = 3 * f[0, 0] - 3 * f[1, 1] + f[2, 2]
    tr = 3 * f[0, -1] - 3 * f[1, -2] + f[2, -3]
    bl = 3 * f[-1, 0] - 3 * f[-2, 1] + f[-3, 2]
    br = 3 * f[-1, -1] - 3 * f[-2, -2] + f[-3, -3]

    return _assemble_padded(interior=f, top=top, bottom=bottom, left=left, right=right,
                            top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)


@tf.function
def pad_constant(n: int, f):
    """Pad 2D tensor by `n` grid units, by copying the nearest value from the edges.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_constant_one(f)  # triggers retracing at each enlargement if we don't use `reduce_retracing=True`
    return f

@tf.function
def pad_linear(n: int, f):
    """Pad 2D tensor by `n` grid units, by linear extrapolation.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_linear_one(f)
    return f

@tf.function
def pad_quadratic(n: int, f):
    """Pad 2D tensor by `n` grid units, by quadratic extrapolation.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_quadratic_one(f)
    return f


# --------------------------------------------------------------------------------
# Denoising

def denoise(N: int, f, *, preserve_range=False):
    """Attempt to denoise function values data on a 2D meshgrid.

    We use a discrete convolution with the Friedrichs mollifier.
    A continuous version is sometimes used as a differentiation technique:

       Ian Knowles and Robert J. Renka. Methods of numerical differentiation of noisy data.
       Electronic journal of differential equations, Conference 21 (2014), pp. 235-246.

    but we use a simple discrete implementation, and only as a denoising preprocessor.

    As far as the output tensor size is concerned, the edges are handled like `padding="SAME"`
    in a convolution. The data is internally padded by local extrapolation by `N` grid units
    in each direction, to produce a useful (even if slightly less accurate) value also near the edges.

    `N`: neighborhood size parameter (how many grid spacings on each axis)
    `f`: function values in meshgrid format, with equal x and y spacing

    `preserve_range`: Denoising brings the function value closer to its neighborhood average, smoothing
                      out peaks. Thus also global extremal values will be attenuated, causing the data
                      range to contract slightly.

                      This effect becomes large if the same data is denoised in a loop, applying the
                      denoiser several times (at each step, feeding in the previous denoised output).

                      If `preserve_range=True`, we rescale the output to preserve the original global min/max
                      values of `f`. Note that if the noise happens to exaggerate those extrema, this will take
                      the scaling from the exaggerated values, not the real ones (which are in general unknown).
    """
    def friedrichs_mollifier(x, *, eps=0.001):  # not normalized!
        return np.where(np.abs(x) < 1 - eps, np.exp(-1 / (1 - x**2)), 0.)

    offset_X, offset_Y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))  # neighbor offsets (in grid points)
    offset_R = np.sqrt(offset_X**2 + offset_Y**2)  # euclidean
    rmax = np.ceil(np.max(offset_R))  # the grid distance at which we want the mollifier to become zero

    kernel = friedrichs_mollifier(offset_R / rmax)
    kernel = kernel / np.sum(kernel)  # normalize our discrete kernel so it sums to 1 (thus actually making it a discrete mollifier)

    # For best numerical results, remap data into [0, 1] before applying the smoother.
    origmax = tf.math.reduce_max(f)
    origmin = tf.math.reduce_min(f)
    f = (f - origmin) / (origmax - origmin)  # [min, max] -> [0, 1]

    # Make `denoise` behave like `padding="same"`, by padding `f` before applying the convolution.
    # We really need a quality padding, at least linear. Simpler paddings are useless here.
    f = pad_quadratic(N, f)
    # paddings = tf.constant([[N, N], [N, N]])
    # f = tf.pad(f, paddings, "SYMMETRIC")

    f = tf.expand_dims(f, axis=0)  # batch
    f = tf.expand_dims(f, axis=-1)  # channels
    kernel = tf.expand_dims(kernel, axis=-1)  # input channels
    kernel = tf.expand_dims(kernel, axis=-1)  # output channels

    f = tf.nn.convolution(f, kernel, padding="VALID")
    f = tf.squeeze(f, axis=-1)  # channels
    f = tf.squeeze(f, axis=0)  # batch

    if preserve_range:
        outmax = tf.math.reduce_max(f)
        outmin = tf.math.reduce_min(f)
        f = (f - outmin) / (outmax - outmin)  # [ε1, 1 - ε2] -> [0, 1]

    # Undo the temporary scaling:
    f = origmin + (origmax - origmin) * f  # [0, 1] -> [min, max]

    return f.numpy()


# --------------------------------------------------------------------------------
# Meshfree differentiation

# # Idea: Let's extend this to arbitrary local neighborhoods in arbitrary geometries.
# # This general neighborhood building code comes from `examples/wlsqm_example.py` in the `wlsqm` package.
# # To actually run with nonhomogeneous neighborhood sizes on the GPU, we need:
# #   - `hoods_validized`: an extra copy of `hoods` where each `-1` has been replaced with a zero (or any valid index)
# #   - Filtering, with something like `tf.where(tf.not_equal(hoods_original, -1), data[hoods_validized], 0.0)`
# #   - Or, we could consider `tf.ragged`: https://www.tensorflow.org/guide/ragged_tensor
# import scipy.spatial
# def build_neighborhoods(S, *, r=5.0, max_neighbors=120):  # S: [[x0, y0], [x1, y1], ...]
#     tree = scipy.spatial.cKDTree(data=S)  # index for fast searching
#     npoints = len(S)
#     hoods = np.zeros((npoints, max_neighbors), dtype=np.int32) - 1  # hoods[i,k] are the indices of neighbor points of S[i], where -1 means "not present"
#     n_neighbors = np.empty((npoints,), dtype=np.int32)              # number of actually present neighbors in hoods[i,:]
#     for i in range(npoints):
#         idxs = tree.query_ball_point(S[i], r)     # indices k of points S[k] that are neighbors of S[i] at distance <= r (but also including S[i] itself!)
#         idxs = [idx for idx in idxs if idx != i]  # exclude S[i] itself
#         if len(idxs) > max_neighbors:
#             idxs = idxs[:max_neighbors]
#         idxs = np.array(idxs, dtype=np.int32)
#         n_neighbors[i] = len(idxs)
#         hoods[i, :n_neighbors[i]] = idxs
#     return hoods, n_neighbors
# x = np.reshape(X, -1)
# y = np.reshape(Y, -1)
# S = np.stack((x, y), axis=-1)  # [[x0, y0], [x1, y1], ...]
# hoods, n_neighbors = build_neighborhoods(S)

# # Improved version, using ragged tensors.
# # But searching on the CPU is still too slow, so maybe don't do this at 256×256 or larger resolution.
# import scipy.spatial
# def build_neighborhoods(S, *, r=5.0, max_neighbors=None):  # S: [[x0, y0], [x1, y1], ...]
#     tree = scipy.spatial.cKDTree(data=S)  # index for fast searching
#     # hoods = []
#     # for i in range(len(S)):
#     #     if i % 1000 == 0:
#     #         print(f"    {i + 1} of {len(S)}...")
#     #     idxs = tree.query_ball_point(S[i], r, workers=-1)  # indices k of points S[k] that are neighbors of S[i] at distance <= r (but also including S[i] itself!)
#     #     idxs = [idx for idx in idxs if idx != i]           # exclude S[i] itself
#     #     if max_neighbors is not None and len(idxs) > max_neighbors:
#     #         idxs = idxs[:max_neighbors]
#     #     hoods.append(idxs)
#     hoods = tree.query_ball_tree(tree, r)  # same thing; should be faster, but has no parallelization option.
#     return tf.ragged.constant(hoods, dtype=tf.int32)

# TODO: add the 6×6 version that also estimates the unknown function value; can be used as a smoother, too.
#  - Assume the neighbors fk exact (we're least-squaring over them, anyway), and the center value fi as missing.
#  - Then least-squares estimate the center value, too.

# TODO: implement classical central differencing, and compare results. Which method is more accurate on a meshgrid? (Likely wlsqm, because more neighbors.)

# TODO: this is now `padding="valid"`; implement `padding="same"` mode (extrapolate linearly?)
# TODO: Implement another version for arbitrary geometries (data point dependent `A` and `c`). (This version is already fine for meshgrid data.)
coeffs = {"dx": 0, "dy": 1, "dx2": 2, "dxdy": 3, "dy2": 4}
def differentiate(N, X, Y, Z):
    """Fit a 2nd order surrogate polynomial to data values on a meshgrid, to estimate derivatives.

    Note the distance matrix `A` (generated automatically) is 5×5 regardless of `N`, but for large `N`,
    assembly takes longer because there are more contributions to each matrix element.

    Note we lose the edges, like in a convolution with padding="VALID", essentially for the same reason.
    This is mathematically trivial to fix (we should just build `A` and `b` without the missing neighbors,
    or take more neighbors from the existing side - even the sizes of A and b do not change), but the code
    becomes unwieldy, so we haven't done that.

    `N`: neighborhood size parameter (how many grid spacings on each axis)
    `X`, `Y`, `Z`: data in meshgrid format for x, y, and function value, respectively
    """
    # Derivative scaling for numerical stability: x' := x / xscale  ⇒  d/dx → (1 / xscale) d/dx'.
    # Choose xscale so that the magnitudes are near 1. Similarly for y. We use the grid spacing (in raw coordinate space) as the scale.
    xscale = X[0, 1] - X[0, 0]
    yscale = Y[1, 0] - Y[0, 0]

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
        dx = dx / xscale
        dy = dy / yscale
        return np.array([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]).T

    # Since we have a uniform grid in this application, the distance matrix of neighbors for each point is the same,
    # so we need to assemble only one.

    # Generic offset distance stencil for all neighbors.
    iy, ix = N, N  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    neighbors = make_stencil(N)  # [#k, 2]
    dx = X[iy + neighbors[:, 0], ix + neighbors[:, 1]] - X[iy, ix]  # [#k]
    dy = Y[iy + neighbors[:, 0], ix + neighbors[:, 1]] - Y[iy, ix]  # [#k]

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
    interior_multi_to_linear = all_multi_to_linear[N:-N, N:-N]

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
    scale = tf.constant([xscale, yscale, xscale**2, xscale * yscale, yscale**2])
    scale = tf.expand_dims(scale, axis=-1)  # for broadcasting
    df = df / scale

    # # Old reshape code:
    # ny, nx = np.shape(X)
    # df = tf.reshape(df, (5, ny - 2 * N, nx - 2 * N))

    df = tf.reshape(df, (5, *tf.shape(interior_multi_to_linear)))
    return df


# --------------------------------------------------------------------------------
# Usage example

def main():
    # --------------------------------------------------------------------------------
    # Parameters

    N = 5  # neighborhood size parameter (how many grid spacings on each axis) for surrogate fitting
    σ = 0.01  # optional: stdev for simulated i.i.d. gaussian noise in data
    xx = np.linspace(0, np.pi, 512)
    yy = xx

    # If σ > 0, how many times to loop the denoiser.
    # If σ = 0, the denoiser is skipped, and this setting has no effect.
    N_denoise_steps = 50

    # --------------------------------------------------------------------------------
    # Set up an expression to generate test data

    x, y = sy.symbols("x, y")
    # expr = sy.sin(x) * sy.cos(y)
    expr = x**2 + y

    # --------------------------------------------------------------------------------
    # Compute the test data

    print("Setup...")

    # Differentiate symbolically to obtain ground truths for the jacobian and hessian to benchmark against.
    dfdx_expr = sy.diff(expr, x, 1)
    dfdy_expr = sy.diff(expr, y, 1)
    d2fdx2_expr = sy.diff(expr, x, 2)
    d2fdxdy_expr = sy.diff(sy.diff(expr, x, 1), y, 1)
    d2fdy2_expr = sy.diff(expr, y, 2)

    dfdx = sy.lambdify((x, y), dfdx_expr)
    dfdy = sy.lambdify((x, y), dfdy_expr)
    d2fdx2 = sy.lambdify((x, y), d2fdx2_expr)
    d2fdxdy = sy.lambdify((x, y), d2fdxdy_expr)
    d2fdy2 = sy.lambdify((x, y), d2fdy2_expr)

    ground_truth_functions = {"dx": dfdx, "dy": dfdy, "dx2": d2fdx2, "dxdy": d2fdxdy, "dy2": d2fdy2}

    f = sy.lambdify((x, y), expr)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)
    print(f"    Function: {expr}")
    print(f"    Data tensor size: {np.shape(Z)}")
    print(f"    Neighborhood radius: {N} grid units")
    print(f"    Synthetic noise stdev: {σ:0.6g}")
    print(f"    Denoise steps: {N_denoise_steps}")

    # # Test the generic neighborhood builder
    # x = np.reshape(X, -1)
    # y = np.reshape(Y, -1)
    # S = np.stack((x, y), axis=-1)  # [[x0, y0], [x1, y1], ...]
    # hoods = build_neighborhoods(S, r=3.0)

    # --------------------------------------------------------------------------------
    # Simulate noisy input, for testing the denoiser.

    if σ > 0:
        # Corrupt the data with synthetic noise...
        print("Synthetic noise...")
        noise = np.random.normal(loc=0.0, scale=σ, size=np.shape(X))
        Z += noise

        # ...and then attempt to remove the noise.
        print("Denoise...")
        for _ in range(N_denoise_steps):  # applying denoise in a loop allows removing relatively large amounts of noise
            # print(f"    step {_ + 1} of {N_denoise_steps}...")
            Z = denoise(N, Z)
            # X, Y = chop_edges(N, X, Y)

    # --------------------------------------------------------------------------------
    # Compute the derivatives.

    print("Differentiating...")
    dZ = differentiate(N, X, Y, Z)
    X_for_dZ, Y_for_dZ = chop_edges(N, X, Y)  # Each `differentiate` loses `N` grid points at the edges, on each axis.

    # Idea: to improve second derivative quality for noisy data, use only first derivatives from wlsqm, and chain the method, with denoising between differentiations.
    print("Smoothed second derivatives:")
    dzdx = dZ[0, :]
    dzdy = dZ[1, :]
    if σ > 0:
        print("    Denoise first derivatives...")
        for _ in range(N_denoise_steps):
            # print(f"        step {_ + 1} of {N_denoise_steps}...")
            dzdx = denoise(N, dzdx)
            dzdy = denoise(N, dzdy)

    print("    Differentiate denoised first derivatives...")
    ddzdx = differentiate(N, X_for_dZ, Y_for_dZ, dzdx)  # jacobian and hessian of dzdx
    ddzdy = differentiate(N, X_for_dZ, Y_for_dZ, dzdy)  # jacobian and hessian of dzdy
    X_for_dZ2, Y_for_dZ2 = chop_edges(N, X_for_dZ, Y_for_dZ)

    d2zdx2 = ddzdx[0, :]
    d2zdxdy = ddzdx[1, :]
    d2zdydx = ddzdy[0, :]  # with exact input in C2, ∂²f/∂x∂y = ∂²f/∂y∂x; we can use this to improve our approximation of ∂²f/∂x∂y
    d2zdy2 = ddzdy[1, :]
    if σ > 0:
        print("    Denoise output...")
        for _ in range(N_denoise_steps):
            # print(f"        step {_ + 1} of {N_denoise_steps}...")
            d2zdx2 = denoise(N, d2zdx2)
            d2zdxdy = denoise(N, d2zdxdy)
            d2zdydx = denoise(N, d2zdydx)
            d2zdy2 = denoise(N, d2zdy2)

    fig = plt.figure(2)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax1.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdx2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("d2f/dx2")
    ground_truth = ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2)
    max_l1_error = np.max(np.abs(ground_truth - d2zdx2))
    print(f"max absolute l1 error dx2 (smoothed) = {max_l1_error:0.3g}")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    d2cross = (d2zdxdy + d2zdydx) / 2.0
    surf = ax2.plot_surface(X_for_dZ2, Y_for_dZ2, d2cross)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("d2f/dxdy")
    ground_truth = ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2)
    max_l1_error = np.max(np.abs(ground_truth - d2cross))
    print(f"max absolute l1 error dxdy (smoothed) = {max_l1_error:0.3g}")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    surf = ax3.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdy2)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("d2f/dy2")
    ground_truth = ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2)
    max_l1_error = np.max(np.abs(ground_truth - d2zdy2))
    print(f"max absolute l1 error dy2 (smoothed) = {max_l1_error:0.3g}")

    fig.suptitle(f"Local quadratic surrogate fit, smoothed second derivatives, noise σ = {σ:0.3g}")
    link_3d_subplot_cameras(fig, [ax1, ax2, ax3])

    # --------------------------------------------------------------------------------
    # Plot the results

    # https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
    print("Plotting.")
    fig = plt.figure(1)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)  # noqa: F841
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("f")
    all_axes = [ax]
    for idx, key in enumerate(coeffs.keys(), start=2):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        surf = ax.plot_surface(X_for_dZ, Y_for_dZ, dZ[coeffs[key], :, :], linewidth=0, antialiased=False)  # noqa: F841
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(key)
        all_axes.append(ax)

        ground_truth = ground_truth_functions[key](X_for_dZ, Y_for_dZ)
        max_l1_error = np.max(np.abs(ground_truth - dZ[coeffs[key], :, :]))
        print(f"max absolute l1 error {key} = {max_l1_error:0.3g}")
    fig.suptitle(f"Local quadratic surrogate fit, noise σ = {σ:0.3g}")
    link_3d_subplot_cameras(fig, all_axes)

if __name__ == '__main__':
    main()
    plt.show()
