"""Smoothing of 1D and 2D tensors by convolution with the Friedrichs mollifier."""

__all__ = ["smooth_2d",
           "smooth_1d"]

import typing

import numpy as np

import tensorflow as tf

from .padding import pad_quadratic_2d, pad_quadratic_1d


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


def smooth_2d(N: int,
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


def smooth_1d(N: int,
              f: typing.Union[np.array, tf.Tensor],
              *,
              padding: str,
              preserve_range: bool = False) -> np.array:
    """Like `smooth_2d`, but for 1D `f`."""
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
