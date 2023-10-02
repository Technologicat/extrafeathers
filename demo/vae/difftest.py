#!/usr/bin/env python
"""Use the weighted least squares meshfree (WLSQM) method to fit a local quadratic surrogate to compute derivatives.

This method is also known as MLS, "moving least squares". The "moving" in the name is an anachronism; a classical
serial CPU implementation would solve the local systems one by one, thus explicitly "moving" the neighborhood.

This method produces approximations for both the jacobian and the hessian in one go, by solving local 5×5 equation systems.

We also provide a version that approximates the function value, the jacobian, and the hessian, by solving local 6×6 equation systems.

This is basically a technique demo; this can be GPU-accelerated with TF, so we can use it to evaluate the
spatial derivatives in the PDE residual in a physically informed loss for up to 2nd order PDEs.

A potential issue is that the VAE output might contain some noise, so to be safe, we need a method that can handle noisy input.

We need only the very basics here. A complete Cython implementation of WLSQM, and documentation:
    https://github.com/Technologicat/python-wlsqm
"""

import typing

from unpythonic import timer

import numpy as np
import sympy as sy
import tensorflow as tf

import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------
# Utilities

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
    if N is None or N < 1:
        raise ValueError(f"Must specify N ≥ 1, got {N}")
    neighbors = [[iy, ix] for iy in range(-N, N + 1)
                          for ix in range(-N, N + 1)]  # if not (iy == 0 and ix == 0) ...but including the center point does no harm.
    neighbors = np.array(neighbors, dtype=int)
    return neighbors


# The edges are nonsense with padding="SAME", so we use "VALID", and chop off the edges of X and Y correspondingly.
def chop_edges(N: int, X: tf.Tensor, Y: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    return X[N:-N, N:-N], Y[N:-N, N:-N]


# --------------------------------------------------------------------------------
# Tensor padding by extrapolation

# TODO: Move tensor padding utilities to a new module.

@tf.function
def _assemble_padded_2d(*, interior: tf.Tensor,
                        top: tf.Tensor, bottom: tf.Tensor, left: tf.Tensor, right: tf.Tensor,
                        top_left: float, top_right: float, bottom_left: float, bottom_right: float) -> tf.Tensor:
    """Assemble a padded rank-2 tensor from parts.

    `interior`: rank-2 tensor, [nrows, ncols]; original tensor to which the padding is to be added
    `top`, `bottom`: rank-1 tensors, [ncols] each; padding edge values
    `left`, `right`: rank-1 tensors, [nrows] each; padding edge values
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

@tf.function
def _assemble_padded_1d(*, interior: tf.Tensor, left: float, right: float) -> tf.Tensor:
    """Like `_assemble_padded_2d`, but for 1D tensor.

    The return value is populated as follows::

        L i i i R
    """
    left = tf.expand_dims(left, axis=0)  # [] -> [1]
    right = tf.expand_dims(right, axis=0)
    padded = tf.concat([left, interior, right], axis=0)  # e.g. [256] -> [258]
    return padded


@tf.function(reduce_retracing=True)
def pad_constant_2d_one(f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by one grid unit, by copying the nearest value from the edges.

    `f`: data in meshgrid format.

    This is really "nearest", but "constant" in the sense of the other paddings defined here:
    we estimate the function value as locally constant.

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

    return _assemble_padded_2d(interior=f, top=top, bottom=bottom, left=left, right=right,
                               top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)

@tf.function(reduce_retracing=True)
def pad_constant_1d_one(f: tf.Tensor) -> tf.Tensor:
    """Pad 1D tensor by one grid unit, by copying the nearest value from the edges."""
    left = f[0]
    right = f[-1]
    return _assemble_padded_1d(interior=f, left=left, right=right)

@tf.function(reduce_retracing=True)
def pad_linear_2d_one(f: tf.Tensor) -> tf.Tensor:
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

    return _assemble_padded_2d(interior=f, top=top, bottom=bottom, left=left, right=right,
                               top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)

@tf.function(reduce_retracing=True)
def pad_linear_1d_one(f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by one grid unit, by linear extrapolation."""
    left = 2 * f[0] - f[1]
    right = 2 * f[-1] - f[-2]
    return _assemble_padded_1d(interior=f, left=left, right=right)

@tf.function(reduce_retracing=True)
def pad_quadratic_2d_one(f: tf.Tensor) -> tf.Tensor:
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

    return _assemble_padded_2d(interior=f, top=top, bottom=bottom, left=left, right=right,
                               top_left=tl, top_right=tr, bottom_left=bl, bottom_right=br)

@tf.function(reduce_retracing=True)
def pad_quadratic_1d_one(f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by one grid unit, by quadratic extrapolation."""
    left = 3 * f[0] - 3 * f[1] + f[2]
    right = 3 * f[-1] - 3 * f[-2] + f[-3]
    return _assemble_padded_1d(interior=f, left=left, right=right)

@tf.function
def pad_constant_2d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by `n` grid units, by copying the nearest value from the edges.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_constant_2d_one(f)  # triggers retracing at each enlargement if we don't use `reduce_retracing=True`
    return f

@tf.function
def pad_constant_1d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 1D tensor by `n` grid units, by copying the nearest value from the edges."""
    for _ in range(n):
        f = pad_constant_1d_one(f)
    return f

@tf.function
def pad_linear_2d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by `n` grid units, by linear extrapolation.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_linear_2d_one(f)
    return f

@tf.function
def pad_linear_1d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 1D tensor by `n` grid units, by linear extrapolation."""
    for _ in range(n):
        f = pad_linear_1d_one(f)
    return f

@tf.function
def pad_quadratic_2d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 2D tensor by `n` grid units, by quadratic extrapolation.

    `n`: how many grid units to pad by.
    `f`: data in meshgrid format.
    """
    for _ in range(n):
        f = pad_quadratic_2d_one(f)
    return f

@tf.function
def pad_quadratic_1d(n: int, f: tf.Tensor) -> tf.Tensor:
    """Pad 1D tensor by `n` grid units, by quadratic extrapolation."""
    for _ in range(n):
        f = pad_quadratic_1d_one(f)
    return f


# --------------------------------------------------------------------------------
# Denoising

def friedrichs_mollifier(x: np.array, *, eps: float = 0.001) -> np.array:
    """The Friedrichs mollifier function.

    This is a non-analytic, symmetric bump centered on the origin, that smoothly
    decreases from `exp(-1)` at `x = 0` to zero at `|x| = 1`. It is zero for any
    `|x| ≥ 1`.

    The function is C∞ continuous also at the seam at `|x| = 1`, justifying the
    name "mollifier".

    However, note that we do not normalize the value! (A mollifier, properly,
    also has the property that its total mass is 1; or in other words, it can
    be thought of as a probability distribution.)

    `x`: arraylike, rank-1.
    `eps`: For numerical stability: wherever `|x| ≥ 1 - ε`, we return zero.
    """
    return np.where(np.abs(x) < 1 - eps, np.exp(-1 / (1 - x**2)), 0.)


def friedrichs_smooth_2d(N: int,
                         f: typing.Union[np.array, tf.Tensor],
                         *,
                         padding: str,
                         preserve_range: bool = False) -> np.array:
    """Attempt to denoise function values data on a 2D meshgrid.

    The method is a discrete convolution with the Friedrichs mollifier.
    A continuous version is sometimes used as a differentiation technique:

       Ian Knowles and Robert J. Renka. Methods of numerical differentiation of noisy data.
       Electronic journal of differential equations, Conference 21 (2014), pp. 235-246.

    but we use a simple discrete implementation, and only as a denoising preprocessor.

    `N`: neighborhood size parameter (how many grid spacings on each axis)
    `f`: function values in meshgrid format, with equal x and y spacing

    `padding`: similar to convolution operations, one of:
        "VALID": Operate in the interior only. This chops off `N` points at the edges on each axis.
        "SAME": Preserve data tensor dimensions. Automatically use local extrapolation to estimate
                `f` outside the edges.

    `preserve_range`: Denoising brings the function value closer to its neighborhood average, smoothing
                      out peaks. Thus also global extremal values will be attenuated, causing the data
                      range to contract slightly.

                      This effect becomes large if the same data is denoised in a loop, applying the
                      denoiser several times (at each step, feeding in the previous denoised output).

                      If `preserve_range=True`, we rescale the output to preserve the original global min/max
                      values of `f`. Note that if the noise happens to exaggerate those extrema, this will take
                      the scaling from the exaggerated values, not the real ones (which are in general unknown).
    """
    if padding.upper() not in ("VALID", "SAME"):
        raise ValueError(f"Invalid padding '{padding}'; valid choices: 'VALID', 'SAME'")

    offset_X, offset_Y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))  # neighbor offsets (in grid units)
    offset_R = np.sqrt(offset_X**2 + offset_Y**2)  # euclidean
    rmax = np.ceil(np.max(offset_R))  # the grid distance at which we want the mollifier to become zero

    kernel = friedrichs_mollifier(offset_R / rmax)
    kernel = kernel / np.sum(kernel)  # normalize our discrete kernel so it sums to 1 (thus actually making it a discrete mollifier)

    # For best numerical results, remap data into [0, 1] before applying the smoother.
    origmax = tf.math.reduce_max(f)
    origmin = tf.math.reduce_min(f)
    f = (f - origmin) / (origmax - origmin)  # [min, max] -> [0, 1]

    if padding.upper() == "SAME":
        # We really need a quality padding, at least linear. Simpler paddings are useless here.
        f = pad_quadratic_2d(N, f)

        # # paddings = tf.constant([[N, N], [N, N]])
        # # f = tf.pad(f, paddings, "SYMMETRIC")

        # # Let's try something even fancier. The results seem nonsense, though?
        # for _ in range(N):
        #     f = pad_quadratic_2d_one(f).numpy()
        #     f[0, :] = friedrichs_smooth_1d(N, f[0, :], padding="SAME", preserve_range=preserve_range)
        #     f[-1, :] = friedrichs_smooth_1d(N, f[-1, :], padding="SAME", preserve_range=preserve_range)
        #     f[:, 0] = friedrichs_smooth_1d(N, f[:, 0], padding="SAME", preserve_range=preserve_range)
        #     f[:, -1] = friedrichs_smooth_1d(N, f[:, -1], padding="SAME", preserve_range=preserve_range)

    f = tf.expand_dims(f, axis=0)  # batch
    f = tf.expand_dims(f, axis=-1)  # channels
    kernel = tf.cast(kernel, f.dtype)
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


def friedrichs_smooth_1d(N: int,
                         f: typing.Union[np.array, tf.Tensor],
                         *,
                         padding: str,
                         preserve_range: bool = False) -> np.array:
    """Like `friedrichs_smooth_2d`, but for 1D `f`."""
    offset_X = np.arange(-N, N + 1)  # neighbor offsets (in grid units)
    kernel = friedrichs_mollifier(offset_X / N)

    # For best numerical results, remap data into [0, 1] before applying the smoother.
    origmax = tf.math.reduce_max(f)
    origmin = tf.math.reduce_min(f)
    f = (f - origmin) / (origmax - origmin)  # [min, max] -> [0, 1]

    if padding.upper() == "SAME":
        f = pad_quadratic_1d(N, f)

    f = tf.expand_dims(f, axis=0)  # batch
    f = tf.expand_dims(f, axis=-1)  # channels
    kernel = tf.cast(kernel, f.dtype)
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
# Meshfree utilities

# In practice, these turned out too slow for lookups over all pixels. Might be possible to fix by constructing only the edge and corner neighbor sets here,
# but then we need different differentiation implementations for the interior (which is fine with the uniform meshgrid implementation) and the edges.

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


# --------------------------------------------------------------------------------
# The WLSQM differentiator, on a meshgrid.

# TODO: implement classical central differencing, and compare results. Which method is more accurate on a meshgrid? (Likely wlsqm, because more neighbors.)

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

def linear_to_multi(idx: tf.Tensor, *, shape: tf.Tensor):  # TODO: we don't actually need this function?
    """[DOF mapper] Convert linear index to meshgrid multi-index.

    We assume C storage order (column index changes fastest).

    `idx`: rank-1 tensor of linear indices, `[idx0, idx1, ...]`.
           Index offsets are fine, too.
    `shape`: rank-1 tensor, containing [ny, nx].

    Returns: rank-2 tensor of meshgrid multi-indices, `[[iy0, ix0], [iy1, ix1], ...]`.
             If input was offsets, returns offsets.
    """
    nx = int(shape[1])
    iy, ix = tf.experimental.numpy.divmod(idx, nx)  # TF 2.13
    return tf.stack([iy, ix], axis=1)  # -> [[iy0, ix0], [iy1, ix1], ...]

def prepare(N: int,
            X: typing.Union[np.array, tf.Tensor],
            Y: typing.Union[np.array, tf.Tensor],
            Z: typing.Union[np.array, tf.Tensor],
            *,
            dtype: tf.DType = tf.float32):
    """Prepare for differentiation on a meshgrid.

    This is the hifiest algorithm provided in this module.

    This function precomputes the surrogate fitting coefficient tensor `c`, and the pixelwise `A` matrices.
    Some of the preparation is performed on the CPU, the most intensive computations on the GPU.

    NOTE: This takes a lot of VRAM. A 6 GB GPU is able to do 256×256, but not much more.

    `N`: Neighborhood size parameter (how many grid spacings on each axis)
         The edges and corners of the input image are handled by clipping the stencil to the data region
         (to use as much data for each pixel as exists within distance `N` on each axis).

    `X`, `Y`, `Z`: data in meshgrid format for x, y, and function value, respectively.
                   The shapes of `X`, `Y` and `Z` must match.

                   `Z` is only consulted for its shape and dtype.

                   The grid spacing must be uniform.

    `dtype`: The desired TensorFlow data type for the outputs `A`, `c`, and `scale`.
             The output `neighbors` always has dtype `int32`.

    The return value is a tuple of tensors, `(A, c, scale, neighbors)`.
    These tensors can be passed to `solve`, which runs completely on the GPU.

    As long as `N`, `X` and `Y` remain constant, and the dtype of `Z` remains the same,
    the same preparation can be reused.
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

    def intarray(x):
        return np.array(x, dtype=int)  # TODO: `np.int32`?

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
                                          for ix in range(-N, N + 1)])  # multi-index offsets
    interior_stencil = multi_to_linear(interior_stencil, shape=shape)  # corresponding linear index offsets
    register_stencil(interior_stencil, interior_idx)

    # Top edge - one stencil per row (N of them, so typically 8).
    for row in range(N):
        top_multi_to_linear = all_multi_to_linear[row, N:-N]
        top_idx = tf.reshape(top_multi_to_linear, [-1])
        top_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                         for ix in range(-N, N + 1)])
        top_stencil = multi_to_linear(top_stencil, shape=shape)
        register_stencil(top_stencil, top_idx)  # each row near the top gets its own stencil

    # Bottom edge - one stencil per row.
    for row in range(-N, 0):
        bottom_multi_to_linear = all_multi_to_linear[row, N:-N]
        bottom_idx = tf.reshape(bottom_multi_to_linear, [-1])
        bottom_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                            for ix in range(-N, N + 1)])
        bottom_stencil = multi_to_linear(bottom_stencil, shape=shape)
        register_stencil(bottom_stencil, bottom_idx)

    # Left edge - one stencil per column (N of them, so typically 8).
    for col in range(N):
        left_multi_to_linear = all_multi_to_linear[N:-N, col]
        left_idx = tf.reshape(left_multi_to_linear, [-1])
        left_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-col, N + 1)])
        left_stencil = multi_to_linear(left_stencil, shape=shape)
        register_stencil(left_stencil, left_idx)

    # Right edge - one stencil per column.
    for col in range(-N, 0):
        right_multi_to_linear = all_multi_to_linear[N:-N, col]
        right_idx = tf.reshape(right_multi_to_linear, [-1])
        right_stencil = intarray([[iy, ix] for iy in range(-N, N + 1)
                                           for ix in range(-N, -col)])
        right_stencil = multi_to_linear(right_stencil, shape=shape)
        register_stencil(right_stencil, right_idx)

    # Upper left corner - one stencil per pixel (N² of them, so typically 64).
    for row in range(N):
        for col in range(N):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])  # just one pixel, but for uniform data format, use a rank-1 tensor
            this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                              for ix in range(-col, N + 1)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Upper right corner - one stencil per pixel.
    for row in range(N):
        for col in range(-N, 0):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-row, N + 1)
                                              for ix in range(-N, -col)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Lower left corner - one stencil per pixel.
    for row in range(-N, 0):
        for col in range(N):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                              for ix in range(-col, N + 1)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    # Lower right corner - one stencil per pixel.
    for row in range(-N, 0):
        for col in range(-N, 0):
            this_idx = tf.constant([all_multi_to_linear[row, col].numpy()])
            this_stencil = intarray([[iy, ix] for iy in range(-N, -row)
                                              for ix in range(-N, -col)])
            this_stencil = multi_to_linear(this_stencil, shape=shape)
            register_stencil(this_stencil, this_idx)

    assert len(stencils) == 1 + 4 * N + 4 * N**2  # interior, edges, corners
    assert not (indirect == -1).any()  # every pixel of the input image should now have a stencil associated with it

    # For meshgrid use, we can store stencils as lists of linear index offsets (int32), in a ragged tensor.
    # Ragged, because points near edges or corners have a clipped stencil with fewer neighbors.
    indirect = tf.constant(indirect, dtype=tf.int32)
    stencils = tf.ragged.constant(stencils, dtype=tf.int32)

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
        dx = dx / xscale  # LHS: offset in scaled space
        dy = dy / yscale
        one = tf.ones_like(dx)  # for the constant term of the fit
        return tf.stack([one, dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], axis=-1)

    # Compute distance of all neighbors (in stencil) for each pixel
    #
    X = tf.reshape(X, [-1])
    Y = tf.reshape(Y, [-1])
    # TODO: Work smarter: we lose our VRAM savings here. But can we do better than either this (too much memory), or Python-loop over pixels (too slow)?
    #
    # `neighbors`: linear indices (not offsets!) of neighbors (in stencil) for each pixel; resolution² * (2 * N + 1)² * 4 bytes, potentially gigabytes of data.
    #              The first term is the base linear index of each data point; the second is the linear index offset of each of its neighbors.
    neighbors = tf.expand_dims(tf.range(npoints), axis=-1) + tf.gather(stencils, indirect)

    # `dx[n, k]`: signed x distance of neighbor `k` from data point `n`. Similarly for `dy[n, k]`.
    dx = tf.gather(X, neighbors) - tf.expand_dims(X, axis=-1)  # `expand_dims` explicitly, to broadcast on the correct axis
    dy = tf.gather(Y, neighbors) - tf.expand_dims(Y, axis=-1)

    # Finally, the surrogate fitting coefficient tensor is:
    c = cnki(dx, dy)

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
    # `einsum` doesn't support `RaggedTensor`, and neither does `tensordot`. What we want to do:
    # # A = tf.einsum("nki,nkj->nij", c, c)
    # # A = tf.tensordot(c, c, axes=[[1], [1]])  # alternative expression
    # Doing it manually:
    rows = []
    for i in range(6):
        row = []
        for j in range(6):
            ci = c[:, :, i]  # -> [#n, #k], where #k is ragged (number of neighbors in stencil for pixel `n`)
            cj = c[:, :, j]  # -> [#n, #k]
            cicj = ci * cj  # [#n, #k]
            Aij = tf.reduce_sum(cicj, axis=1)  # -> [#n]
            row.append(Aij)
        row = tf.stack(row, axis=1)  # -> [#n, #cols]
        rows.append(row)
    A = tf.stack(rows, axis=1)  # [[#n, #cols], [#n, #cols], ...] -> [#n, #rows, #cols]

    # # DEBUG: If the x and y scalings work, the range of values in `A` should be approximately [0, (2 * N + 1)²].
    # # The upper bound comes from the maximal number of points in the stencil, and is reached when gathering this many "ones" in the constant term.
    # absA = tf.abs(A)
    # print(f"A[n, i, j] ∈ [{tf.reduce_min(absA):0.6g}, {tf.reduce_max(absA):0.6g}]")
    # print(A)

    # TODO: DEBUG: sanity check that each `A[n, :, :]` is symmetric.

    # Scaling factors to undo the derivative scaling,  d/dx' → d/dx.  `solve` needs this to postprocess its results.
    scale = tf.constant([1.0, xscale, yscale, xscale**2, xscale * yscale, yscale**2], dtype=Z.dtype)
    # scale = tf.expand_dims(scale, axis=-1)  # for broadcasting; solution shape from `tf.linalg.solve` is [6, npoints]

    A = tf.cast(A, dtype)
    c = tf.cast(c, dtype)
    scale = tf.cast(scale, dtype)

    return A, c, scale, neighbors

@tf.function
def solve(a: tf.Tensor,
          c: tf.Tensor,
          scale: tf.Tensor,
          neighbors: tf.RaggedTensor,
          z: tf.Tensor):
    """[kernel] Assemble and solve system that was prepared using `prepare`.

    This function runs completely on the GPU, and is differentiable, so it can be used e.g. inside a neural network loss function.
    """
    shape = tf.shape(z)
    z = tf.reshape(z, [-1])
    z = tf.cast(z, a.dtype)

    absz = tf.abs(z)
    zmin = tf.reduce_min(absz)
    zmax = tf.reduce_max(absz)
    z = (z - zmin) / (zmax - zmin)  # -> [0, 1]

    # b[n,i] = ∑k( z[neighbors[n,k]] * c[n,k,i] )
    rows = []
    for i in range(6):
        zgnk = tf.gather(z, neighbors)  # -> [#n, #k], ragged in k
        ci = c[:, :, i]  # -> [#n, #k]
        zci = zgnk * ci  # [#n, #k]
        bi = tf.reduce_sum(zci, axis=1)  # -> [#n]
        bi = tf.cast(bi, a.dtype)
        rows.append(bi)
    b = tf.stack(rows, axis=1)  # -> [#n, #rows]
    b = tf.expand_dims(b, axis=-1)  # -> [#n, #rows, 1]  (in this variant of the algorithm, we have just one RHS for each LHS matrix)

    sol = tf.linalg.solve(a, b)  # -> [#n, #rows, 1]
    # print(tf.shape(sol))  # [#n, #rows, 1]
    # print(tf.math.reduce_max(abs(tf.matmul(a, sol) - b)))  # DEBUG: yes, the solutions are correct.
    sol = tf.squeeze(sol, axis=-1)  # -> [#n, #rows]

    sol = sol / scale  # return derivatives from scaled x, y (as set up by `prepare`) to raw x, y
    sol = zmin + (zmax - zmin) * sol  # return from scaled z to raw z

    sol = tf.transpose(sol, [1, 0])  # -> [#rows, #n]
    return tf.reshape(sol, (6, int(shape[0]), int(shape[1])))  # -> [#rows, ny, nx]


# See `wlsqm.pdf` in the `python-wlsqm` docs for details on the algorithm.
coeffs_diffonly = {"dx": 0, "dy": 1, "dx2": 2, "dxdy": 3, "dy2": 4}

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
    neighbors = stencil if stencil is not None else make_stencil(N)  # [#k, 2]
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
# Generalized WLSQM (fit also function values), on a meshgrid.
#
# See `wlsqm_gen.pdf` in the `python-wlsqm` docs for details on the algorithm.

coeffs_full = {"f": 0, "dx": 1, "dy": 2, "dx2": 3, "dxdy": 4, "dy2": 5}

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

    neighbors = stencil if stencil is not None else make_stencil(N)  # [#k, 2]
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
    # bs = tf.einsum("nk,ki->ni", fgnk, c)  # This would be clearer...
    bs = tf.einsum("nk,ki->in", fgnk, c)  # ...but this is the ordering `tf.linalg.solve` wants (components on axis 0, batch on axis 1).

    # The solution of the linear systems (one per data point) yields the jacobian and hessian of the surrogate.
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

    neighbors = stencil if stencil is not None else make_stencil(N)  # [#k, 2]
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
    # bs = tf.einsum("nk,ki->ni", fgnk, c)  # This would be clearer...
    bs = tf.einsum("nk,ki->in", fgnk, c)  # ...but this is the ordering `tf.linalg.solve` wants (components on axis 0, batch on axis 1).

    # The solution of the linear systems (one per data point) yields the jacobian and hessian of the surrogate.
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

    neighbors = stencil if stencil is not None else make_stencil(N)  # [#k, 2]
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
    # bs = tf.einsum("nk,ki->ni", fgnk, c)  # This would be clearer...
    bs = tf.einsum("nk,ki->in", fgnk, c)  # ...but this is the ordering `tf.linalg.solve` wants (components on axis 0, batch on axis 1).

    # The solution of the linear systems (one per data point) yields the jacobian and hessian of the surrogate.
    df = tf.linalg.solve(A, bs)  # [1, n_interior_points]

    df = tf.reshape(df, (1, *tf.shape(interior_multi_to_linear)))
    return df * zscale


# --------------------------------------------------------------------------------
# Differentiation with improved edge handling

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


# --------------------------------------------------------------------------------
# Usage example

def main():
    # --------------------------------------------------------------------------------
    # Parameters

    # `N`: neighborhood size parameter (radius in grid units) for surrogate fitting
    # `σ`: optional (set to 0 to disable): stdev for simulated i.i.d. gaussian noise in data

    # Currently, 8 is the largest numerically stable neighborhood size, and yields the best results for noisy data.
    N, σ = 8, 0.001

    # # 3 seems enough when the data is numerically exact.
    # N, σ = 3, 0.0

    # This demo seems to yield best results (least l1 error) at 256 pixels per axis.
    #
    # This is still horribly slow despite GPU acceleration. Performance is currently CPU-bound.
    # Denoising is the performance bottleneck. With numerically exact data, differentiation is acceptably fast.
    #
    # Even at 512 resolution, GPU utilization is under 20% (according to `nvtop`), and there is barely any noticeable difference
    # in the surrogate fitting speed. 512 is the largest that works, at least on Quadro RTX 3000 mobile (RTX 2xxx based chip, 6 GB VRAM).
    #
    # At 768 or 1024, cuBLAS errors out (cuBlas call failed status = 14 [Op:MatrixSolve]).
    # Currently I don't know why - there should be no difference other than the batch size (the whole image is sent in one batch).
    # Solving a linear system with 1024 unknowns should hardly take gigabytes of VRAM even at float32.
    resolution = 256

    # If σ > 0, how many times to loop the denoiser.
    # If σ = 0, denoising is skipped, and this setting has no effect.
    #
    # For N = 8, σ = 0.001, resolution = 256, it seems 10 steps is the smallest number that yields acceptable results.
    denoise_steps = 10

    # --------------------------------------------------------------------------------
    # Set up an expression to generate test data

    x, y = sy.symbols("x, y")
    expr = sy.sin(x) * sy.cos(y)
    # expr = x**2 + y

    # --------------------------------------------------------------------------------
    # Compute the test data

    print("Setup...")

    with timer() as tim:
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

        xx = np.linspace(0, np.pi, resolution)
        yy = xx
        f = sy.lambdify((x, y), expr)
        ground_truth_functions = {"f": f, "dx": dfdx, "dy": dfdy, "dx2": d2fdx2, "dxdy": d2fdxdy, "dy2": d2fdy2}

        X, Y = np.meshgrid(xx, yy)
        Z = f(X, Y)
    print(f"    Done in {tim.dt:0.6g}s.")

    print(f"    Function: {expr}")
    print(f"    Data tensor size: {np.shape(Z)}")
    print(f"    Neighborhood radius: {N} grid units (neighborhood size {2 * N + 1}×{2 * N + 1} = {(2 * N + 1)**2} grid points)")
    if σ > 0:
        print(f"    Synthetic noise stdev: {σ:0.6g}")
        print(f"    Denoise steps: {denoise_steps}")
    else:
        print("    No synthetic noise")

    # # Test the generic neighborhood builder
    # x = np.reshape(X, -1)
    # y = np.reshape(Y, -1)
    # S = np.stack((x, y), axis=-1)  # [[x0, y0], [x1, y1], ...]
    # hoods = build_neighborhoods(S, r=3.0)

    # --------------------------------------------------------------------------------
    # Simulate noisy input, for testing the denoiser.

    preps = prepare(N, X, Y, Z)

    def denoise(N, X, Y, Z):
        # denoise by least squares
        for _ in range(denoise_steps):
            print(f"    Least squares fitting: step {_ + 1} of {denoise_steps}...")
            # tmp = hifier_differentiate(N, X, Y, Z, kernel=fit_quadratic)
            tmp = solve(*preps, Z)
            Z = tmp[coeffs_full["f"]]

        # denoise by Friedrichs smoothing
        for _ in range(denoise_steps):  # applying denoise in a loop allows removing relatively large amounts of noise
            print(f"    Friedrichs smoother: step {_ + 1} of {denoise_steps}...")
            Z = friedrichs_smooth_2d(N, Z, padding="SAME")
            # X, Y = chop_edges(N, X, Y)

        return Z

    if σ > 0:
        # Corrupt the data with synthetic noise.
        print("Add synthetic noise...")
        with timer() as tim:
            noise = np.random.normal(loc=0.0, scale=σ, size=np.shape(X))
            Z += noise
        print(f"    Done in {tim.dt:0.6g}s.")

        # Estimate the amount of noise.
        # Note we only need the noisy data to compute the estimate. We do this by least-squares fitting the function values.
        #
        # We see the estimate works pretty well - the detected RMS noise level is approximately the stdev of the gaussian synthetic noise,
        # matching the true noise level.
        print("Test: estimate noise level...")
        with timer() as tim:
            tmp = hifier_differentiate(N, X, Y, Z, kernel=fit_quadratic)
            noise_estimate = Z - tmp[coeffs_full["f"], :]
            del tmp
            estimated_noise_RMS = np.mean(noise_estimate**2)**0.5
            true_noise_RMS = np.mean(noise**2)**0.5  # ground truth
        print(f"    Done in {tim.dt:0.6g}s.")
        print(f"    Noise (RMS): estimated {estimated_noise_RMS:0.6g}, true {true_noise_RMS:0.6g}")

        # ...and then attempt to remove the noise.
        print("Denoise function values...")
        with timer() as tim:
            Z = denoise(N, X, Y, Z)
        print(f"    Done in {tim.dt:0.6g}s.")

    # --------------------------------------------------------------------------------
    # Compute the derivatives.

    print("Differentiate...")
    with timer() as tim:
        dZ = solve(*preps, Z)[1:, :]
        # dZ = hifier_differentiate(N, X, Y, Z)
        # X_for_dZ, Y_for_dZ = chop_edges(N, X, Y)  # Each `differentiate` in `padding="VALID"` mode loses `N` grid points at the edges, on each axis.
        X_for_dZ, Y_for_dZ = X, Y  # In `padding="SAME"` mode, the dimensions are preserved, but the result may not be accurate near the edges.
    print(f"    Done in {tim.dt:0.6g}s.")

    # # Just denoising the second derivatives doesn't improve their quality much.
    # d2zdx2 = dZ[coeffs_diffonly["dx2"], :]
    # d2zdxdy = dZ[coeffs_diffonly["dxdy"], :]
    # d2zdy2 = dZ[coeffs_diffonly["dy2"], :]
    # X_for_dZ2, Y_for_dZ2 = X_for_dZ, Y_for_dZ
    # if σ > 0:
    #     print("    Denoise second derivatives...")
    #     with timer() as tim:
    #         d2zdx2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdx2)
    #         d2zdxdy = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdxdy)
    #         d2zdy2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdy2)
    #     print(f"        Done in {tim.dt:0.6g}s.")
    # d2cross = d2zdxdy

    # To improve second derivative quality for noisy data, we can first compute first derivatives by wlsqm,
    # and then chain the method, with denoising between differentiations. In this variant, a final denoising
    # step also helps.
    print("Smoothed second derivatives:")
    dzdx = dZ[coeffs_diffonly["dx"], :]
    dzdy = dZ[coeffs_diffonly["dy"], :]
    if σ > 0:
        print("    Denoise first derivatives...")
        with timer() as tim:
            dzdx = denoise(N, X_for_dZ, Y_for_dZ, dzdx)
            dzdy = denoise(N, X_for_dZ, Y_for_dZ, dzdy)
        print(f"        Done in {tim.dt:0.6g}s.")

    print("    Differentiate denoised first derivatives...")
    with timer() as tim:
        ddzdx = solve(*preps, dzdx)[1:, :]
        ddzdy = solve(*preps, dzdy)[1:, :]
        # ddzdx = hifier_differentiate(N, X_for_dZ, Y_for_dZ, dzdx)  # jacobian and hessian of dzdx
        # ddzdy = hifier_differentiate(N, X_for_dZ, Y_for_dZ, dzdy)  # jacobian and hessian of dzdy
        # X_for_dZ2, Y_for_dZ2 = chop_edges(N, X_for_dZ, Y_for_dZ)
        X_for_dZ2, Y_for_dZ2 = X_for_dZ, Y_for_dZ
    print(f"        Done in {tim.dt:0.6g}s.")

    d2zdx2 = ddzdx[coeffs_diffonly["dx"], :]
    d2zdxdy = ddzdx[coeffs_diffonly["dy"], :]
    d2zdydx = ddzdy[coeffs_diffonly["dx"], :]  # with exact input in C2, ∂²f/∂x∂y = ∂²f/∂y∂x; we can use this to improve our approximation of ∂²f/∂x∂y
    d2zdy2 = ddzdy[coeffs_diffonly["dy"], :]
    if σ > 0:
        print("    Denoise obtained second derivatives...")
        with timer() as tim:
            d2zdx2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdx2)
            d2zdxdy = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdxdy)
            d2zdydx = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdydx)
            d2zdy2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdy2)
        print(f"        Done in {tim.dt:0.6g}s.")

    d2cross = (d2zdxdy + d2zdydx) / 2.0

    # --------------------------------------------------------------------------------
    # Plot the results

    print("Plotting.")
    with timer() as tim:
        fig = plt.figure(2)
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        surf = ax1.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdx2)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("d2f/dx2")
        ground_truth = ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdx2))
        print(f"    max absolute l1 error dx2 (smoothed) = {max_l1_error:0.3g}")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        surf = ax2.plot_surface(X_for_dZ2, Y_for_dZ2, d2cross)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("d2f/dxdy")
        ground_truth = ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2cross))
        print(f"    max absolute l1 error dxdy (smoothed) = {max_l1_error:0.3g}")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        surf = ax3.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdy2)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_title("d2f/dy2")
        ground_truth = ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdy2))
        print(f"    max absolute l1 error dy2 (smoothed) = {max_l1_error:0.3g}")

        fig.suptitle(f"Local quadratic surrogate fit, smoothed second derivatives, noise σ = {σ:0.3g}")
        link_3d_subplot_cameras(fig, [ax1, ax2, ax3])

        # https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
        fig = plt.figure(1)
        ax = fig.add_subplot(2, 3, 1, projection="3d")
        surf = ax.plot_surface(X, Y, Z)  # noqa: F841
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("f")

        ground_truth = ground_truth_functions["f"](X, Y)
        max_l1_error = np.max(np.abs(Z - ground_truth))
        print(f"    max absolute l1 error f = {max_l1_error:0.3g} (from denoising)")

        all_axes = [ax]
        for idx, key in enumerate(coeffs_diffonly.keys(), start=2):
            ax = fig.add_subplot(2, 3, idx, projection="3d")
            surf = ax.plot_surface(X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly[key], :, :])  # noqa: F841  # , linewidth=0, antialiased=False
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(key)
            all_axes.append(ax)

            ground_truth = ground_truth_functions[key](X_for_dZ, Y_for_dZ)
            max_l1_error = np.max(np.abs(dZ[coeffs_diffonly[key], :, :] - ground_truth))
            print(f"    max absolute l1 error {key} = {max_l1_error:0.3g}")
        fig.suptitle(f"Local quadratic surrogate fit, noise σ = {σ:0.3g}")
        link_3d_subplot_cameras(fig, all_axes)

        # l1 errors
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        def plot_one(ax, X, Y, Z, title):
            L, U = np.min(Z), np.max(Z)
            v = max(abs(L), abs(U))
            theplot = ax.pcolormesh(X, Y, Z, vmin=-v, vmax=v, cmap="RdBu_r")
            fig.colorbar(theplot, ax=ax)
            ax.set_aspect("equal")
            ax.set_title(title)
        plot_one(axs[0, 0], X, Y, Z - ground_truth_functions["f"](X, Y), "f")
        plot_one(axs[0, 1], X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly["dx"]] - ground_truth_functions["dx"](X_for_dZ, Y_for_dZ), "dx")
        plot_one(axs[0, 2], X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly["dy"]] - ground_truth_functions["dy"](X_for_dZ, Y_for_dZ), "dy")
        plot_one(axs[1, 0], X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly["dx2"]] - ground_truth_functions["dx2"](X_for_dZ, Y_for_dZ), "dx2")
        plot_one(axs[1, 1], X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly["dxdy"]] - ground_truth_functions["dxdy"](X_for_dZ, Y_for_dZ), "dxdy")
        plot_one(axs[1, 2], X_for_dZ, Y_for_dZ, dZ[coeffs_diffonly["dy2"]] - ground_truth_functions["dy2"](X_for_dZ, Y_for_dZ), "dy2")
        plot_one(axs[2, 0], X_for_dZ2, Y_for_dZ2, d2zdx2 - ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2), "dx2 (smoothed)")
        plot_one(axs[2, 1], X_for_dZ2, Y_for_dZ2, d2cross - ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2), "dxdy (smoothed)")
        plot_one(axs[2, 2], X_for_dZ2, Y_for_dZ2, d2zdy2 - ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2), "dy2 (smoothed)")
        fig.suptitle("l1 error (fitted - ground truth)")
    print(f"    Done in {tim.dt:0.6g}s.")

if __name__ == '__main__':
    main()
    plt.show()
