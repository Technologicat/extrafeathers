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

import math
import typing

from unpythonic import timer

import numpy as np
import sympy as sy
import tensorflow as tf

import matplotlib.pyplot as plt

from .differentiate import prepare, solve_lu, coeffs_full

# TODO: implement also classical central differencing, and compare results. Which method is more accurate on a meshgrid? (Likely wlsqm, because many more neighbors.)

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


# The edges are nonsense with padding="SAME", so we use "VALID", and chop off the edges of X and Y correspondingly.
def chop_edges(N: int, X: tf.Tensor, Y: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    return X[N:-N, N:-N], Y[N:-N, N:-N]

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
# Usage example

def main():
    # # xgesv.py debugging/testing
    # # tf.debugging.set_log_device_placement(True)
    # from .xgesv import (decompose, solve as linsolve, unpack,
    #                     decompose_one, solve_one as linsolve_one, unpack_one)
    #
    # A = tf.constant([[4, 1, 0, 0, 2],
    #                  [3, 3, -1, -1, 0],
    #                  [0, 0, 1, 1, 1],
    #                  [-2, 1, -2, 3, 7],
    #                  [0, 0, 2, 2, 1]], dtype=tf.float16)
    # b = tf.constant([19, 6, 8, 31, 11], dtype=tf.float16)
    # print("-" * 80)
    # print("Inputs")
    # print(A)
    # print(b)
    #
    # def test_single():
    #     print("-" * 80)
    #     print("Single test")
    #     LU, p = decompose_one(A)
    #     x = linsolve_one(LU, p, b)
    #     L, U = unpack_one(LU)
    #     print(L)
    #     print(U)
    #     print(p)
    #     print(x)
    #     print("Decomposition residual L U - P A")
    #     P = tf.linalg.LinearOperatorPermutation(p, dtype=LU.dtype, name="P")
    #     # print(tf.tensordot(L, U, axes=[[1], [0]]))  # L U
    #     print(tf.linalg.matmul(L, U) - P.matmul(A))
    #     print("Equation system residual A x - b")
    #     print(tf.linalg.matvec(A, x) - b)
    # test_single()
    #
    # A = tf.expand_dims(A, axis=0)
    # b = tf.expand_dims(b, axis=0)
    # def test_batched():
    #     print("-" * 80)
    #     print("Batched test")
    #     LU, p = decompose(A)
    #     x = linsolve(LU, p, b)
    #     L, U = unpack(LU)
    #     print(L)
    #     print(U)
    #     print(p)
    #     print(x)
    #     print("Decomposition residual L U - P A")
    #     P = tf.linalg.LinearOperatorPermutation(p, dtype=LU.dtype, name="P")
    #     print(tf.linalg.matmul(L, U) - P.matmul(A))
    #     print("Equation system residual A x - b")
    #     print(tf.linalg.matvec(A, x) - b)
    # test_batched()

    # --------------------------------------------------------------------------------
    # Parameters

    # Input image resolution (in pixels per axis; the image is square).
    #
    # This demo seems to yield best results (least l1 error) at 256.
    #
    # The hifiest algorithm (`prepare`/`solve`) works at least up to 1024 on a 6 GB card.
    resolution = 256

    # `N`: Neighborhood size parameter for surrogate fitting. The data used for fitting the local
    #      surrogate model for each pixel consists of a box of [2 * N + 1, 2 * N + 1] pixels,
    #      centered on the pixel being fitted. For pixels near edges and corners, the box is
    #      clipped to the data region. Furthermore, if the p-norm setting is other than "inf",
    #      the box is filtered to keep only points that are within a p-norm distance of `N`
    #      from the center.
    #
    #      Note the surrogate modeling assumption: a quadratic polynomial should be able to
    #      reasonably describe the function locally in each neighborhood.
    #
    # `σ`: Optional, set to 0 to disable: stdev for per-pixel synthetic noise (i.i.d. gaussian).

    # `p` for the p-norm, for determining neighborhood shape (see `prepare`).
    # Either float >= 1.0, or the string "inf" (to use the whole box).
    # Especially useful is `p=2.0`, i.e. the Euclidean norm, creating a round neighborhood.
    p = 2.0

    # Pixel count (i.e. stencil size) as a function of neighborhood size parameter `N`:
    #
    #    N   box (p="inf")  Euclidean (p=2.0)
    # ---------------------------------------
    #    1    3² =  9         -
    #    2    5² =  25       13
    #    3    7² =  49       29
    #    4    9² =  81       49
    #    5   11² = 121       81
    #    6   13² = 169      113
    #    7   15² = 225      149
    #    8   17² = 289      197
    #    9   19² = 361      253
    #   10   21² = 441      317
    #   11   23² = 529      377
    #   12   25² = 625      441
    #   13   27² = 729      529
    #
    # The `TF_GPU_ALLOCATOR=cuda_malloc_async` environment variable may allow using
    # slightly larger stencils than without it.
    #
    # At `N = 1`, Euclidean neighborhoods would have 5 points, but the surrogate fitting
    # algorithm needs at least 7 to make its system matrix invertible.
    #
    # Note the count is always odd, because every other pixel has a symmetric pair,
    # but the pixel at the center doesn't.
    #
    # In practice, with `p=2.0`, it is better to use a half-integer `N`, to avoid
    # one pixel sticking out from the otherwise smooth circle in each cardinal direction.
    # Here are pixel counts as a function of half-integer `N`, up to 40.5:
    #
    #    N   Euclidean (p=2.0)
    # ------------------------
    #  1.5      9 (not invertible; 3×3 box, not enough unique dx and dy values to detect dx2, dy2)
    #  2.5     21
    #  3.5     37
    #  4.5     69
    #  5.5     97
    #  6.5    137
    #  7.5    177
    #  8.5    225
    #  9.5    293
    # 10.5    349
    # 11.5    421
    # 12.5    489
    # 13.5    577
    # 14.5    665
    # 15.5    749
    # 16.5    861
    # 17.5    973
    # ----------- 1k neighbors (1024)
    # 18.5   1085
    # 19.5   1201
    # 20.5   1313
    # 21.5   1457
    # 22.5   1597
    # 23.5   1741
    # 24.5   1885
    # ----------- 2k neighbors (2048)
    # 25.5   2053
    # 26.5   2217
    # 27.5   2377
    # 28.5   2561
    # 29.5   2733
    # 30.5   2933
    # ----------- 3k neighbors (3072)
    # 31.5   3125
    # 32.5   3313
    # 33.5   3521
    # 34.5   3745
    # 35.5   3969
    # ----------- 4k neighbors (4096)
    # 36.5   4197
    # 37.5   4421
    # 38.5   4669
    # 39.5   4905
    # ----------- 5k neighbors (5120)
    # 40.5   5169
    # 41.5   5417
    # 42.5   5681
    # 43.5   5957 [VRAM OOM at 6 GB]
    #
    # N, σ = 2.5, 0.0
    # N, σ = 25.5, 0.001
    N, σ = 40.5, 0.01
    N_int = math.ceil(N)

    # # A very small stencil seems enough for good results when the data is numerically exact.
    # N, σ = 2.5, 0.0

    # If σ > 0, how many times to loop the denoiser. Larger neighborhood sizes need less denoising.
    # If σ = 0, denoising is skipped, and this setting has no effect.
    #
    # To add synthetic noise, but skip denoising (to see how the results deteriorate),
    # set σ > 0 and `denoise_steps = 0`.
    #
    # If the actual `f` is smooth enough, and if enough VRAM is available, then to eliminate high-frequency noise,
    # it's usually a better deal to just increase `N` (to average out the noise in a larger area during fitting),
    # rather than to enable denoising.
    #
    # (Denoising essentially makes the patches with smaller `N` communicate between denoise steps,
    #  thus emulating the effects of a larger `N`.)
    denoise_steps = 0

    # Batch size (data points) for system matrix and load vector assembly for low VRAM mode of `prepare` and `solve`.
    batch_size = 8192

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
        # We will use `float32` for GPU computation anyway, so typecasting here saves
        # an expensive extra `@tf.function` tracing for the initial `float64` data.
        Z = tf.cast(Z, dtype=tf.float32)

        # `prepare` only takes shape and dtype from `Z`.
        preps, stencil = prepare(N, X, Y, Z, p=p,
                                 format="LUp",
                                 low_vram=True, low_vram_batch_size=batch_size,
                                 print_statistics=True, indent=" " * 4)
    print(f"    Done in {tim.dt:0.6g}s.")

    print(f"    Function: {expr}")
    print(f"    Data tensor size: {np.shape(Z)}")
    print(f"    Low VRAM batch size: {batch_size} data points per batch (⇒ {math.ceil(np.prod(np.shape(Z)) / batch_size)} batches)")
    print(f"    Neighborhood radius: {N} grid units (p-norm p = {p}; stencil size {len(stencil)} grid points)")
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

    solve_called = False
    def solve(Z):
        nonlocal solve_called
        if not solve_called:
            print("    NOTE: First call to `solve`; tracing solver graph. This may take a while.")
            solve_called = True
        return solve_lu(*preps, Z, low_vram=True, low_vram_batch_size=batch_size)

    def denoise(N, X, Y, Z, *, indent=4):
        # Applying denoising in a loop allows removing larger amounts of noise.
        # Effectively, the neighboring patches communicate between iterations.
        for _ in range(denoise_steps):
            print(f"{indent * ' '}Denoising: step {_ + 1} of {denoise_steps}...")
            # tmp = hifier_differentiate(N, X, Y, Z, kernel=fit_quadratic)
            tmp = solve(Z)  # lsq
            Z = tmp[coeffs_full["f"]]
            # Z = smooth_2d(N, Z, padding="SAME")  # Friedrichs (eliminates high-frequency noise, but highly unstable extrapolation at edges/corners)
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
        print("[For information] Estimate noise level...")
        with timer() as tim:
            # tmp = hifier_differentiate(N, X, Y, Z, kernel=fit_quadratic)
            tmp = solve(Z)
            noise_estimate = Z - tmp[coeffs_full["f"]]
            del tmp
            estimated_noise_RMS = np.mean(noise_estimate**2)**0.5
            true_noise_RMS = np.mean(noise**2)**0.5  # ground truth
        print(f"    Done in {tim.dt:0.6g}s.")
        print(f"    Noise (RMS): estimated {estimated_noise_RMS:0.6g}, true {true_noise_RMS:0.6g}")

    # --------------------------------------------------------------------------------
    # Compute the derivatives.

    with timer() as tim_total:
        # Attempt to remove the noise.
        if σ > 0 and denoise_steps > 0:
            print("Denoise input data...")
            with timer() as tim:
                print("    f...")
                Z = denoise(N, X, Y, Z)
            print(f"    Done in {tim.dt:0.6g}s.")

        print("Differentiate input data...")
        with timer() as tim:
            print("    f...")
            dZ = solve(Z)
            # dZ = hifier_differentiate(N, X, Y, Z)
            # X_for_dZ, Y_for_dZ = chop_edges(N, X, Y)  # Each `differentiate` in `padding="VALID"` mode loses `N` grid points at the edges, on each axis.
            X_for_dZ, Y_for_dZ = X, Y  # In `padding="SAME"` mode, the dimensions are preserved, but the result may not be accurate near the edges.
        print(f"    Done in {tim.dt:0.6g}s.")

        # # Just denoising the second derivatives doesn't improve their quality much.
        # d2zdx2 = dZ[coeffs_full["dx2"]]
        # d2zdxdy = dZ[coeffs_full["dxdy"]]
        # d2zdy2 = dZ[coeffs_full["dy2"]]
        # X_for_dZ2, Y_for_dZ2 = X_for_dZ, Y_for_dZ
        # if σ > 0:
        #     print("    Denoise second derivatives...")
        #     with timer() as tim:
        #         d2zdx2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdx2, indent=8)
        #         d2zdxdy = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdxdy, indent=8)
        #         d2zdy2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdy2, indent=8)
        #     print(f"        Done in {tim.dt:0.6g}s.")
        # d2cross = d2zdxdy

        # To improve second derivative quality for noisy data, we can first compute first derivatives by wlsqm,
        # and then chain the method, with denoising between differentiations. In this variant, a final denoising
        # step also helps.
        #
        # Even without noise, this seems to slightly reduce the maximum l1 fitting error. This is probably
        # because by doing this, we obtain a linear surrogate model for the second derivatives (because we
        # refit a quadratic surrogate model to the first derivatives) - so the model has the same level of
        # capability as the original surrogate has for the first derivatives.
        print("Refit second derivatives:")
        dzdx = dZ[coeffs_full["dx"]]
        dzdy = dZ[coeffs_full["dy"]]
        if σ > 0 and denoise_steps > 0:
            print("    Denoise first derivatives...")
            with timer() as tim:
                print("        dx...")
                dzdx = denoise(N, X_for_dZ, Y_for_dZ, dzdx, indent=8)
                print("        dy...")
                dzdy = denoise(N, X_for_dZ, Y_for_dZ, dzdy, indent=8)
            print(f"        Done in {tim.dt:0.6g}s.")

        print("    Differentiate first derivatives...")
        with timer() as tim:
            print("        dx...")
            ddzdx = solve(dzdx)
            print("        dy...")
            ddzdy = solve(dzdy)
            # ddzdx = hifier_differentiate(N, X_for_dZ, Y_for_dZ, dzdx)  # jacobian and hessian of dzdx
            # ddzdy = hifier_differentiate(N, X_for_dZ, Y_for_dZ, dzdy)  # jacobian and hessian of dzdy
            # X_for_dZ2, Y_for_dZ2 = chop_edges(N, X_for_dZ, Y_for_dZ)
            X_for_dZ2, Y_for_dZ2 = X_for_dZ, Y_for_dZ
        print(f"        Done in {tim.dt:0.6g}s.")

        d2zdx2 = ddzdx[coeffs_full["dx"]]
        d2zdxdy = ddzdx[coeffs_full["dy"]]
        d2zdydx = ddzdy[coeffs_full["dx"]]  # with exact input in C2, ∂²f/∂x∂y = ∂²f/∂y∂x; we can use this to improve our approximation of ∂²f/∂x∂y
        d2zdy2 = ddzdy[coeffs_full["dy"]]
        if σ > 0 and denoise_steps > 0:
            print("    Denoise obtained second derivatives...")
            with timer() as tim:
                print("        dx2...")
                d2zdx2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdx2, indent=8)
                print("        dxdy...")
                d2zdxdy = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdxdy, indent=8)
                print("        dydx...")
                d2zdydx = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdydx, indent=8)
                print("        dy2...")
                d2zdy2 = denoise(N, X_for_dZ2, Y_for_dZ2, d2zdy2, indent=8)
            print(f"        Done in {tim.dt:0.6g}s.")

        d2cross = (d2zdxdy + d2zdydx) / 2.0
    print(f"Total wall time for differentiation {tim_total.dt:0.6g}s.")

    # --------------------------------------------------------------------------------
    # Plot the results

    print("Plotting.")
    with timer() as tim:
        # https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
        fig = plt.figure(1, figsize=(12, 12))

        # NOTE:
        #  Z: original data, with denoising applied
        #  dZ[coeffs_full["f"]]: lsq fitted data (from the first differentiation, even when `denoising_steps=0`)

        # Function itself, and the raw first and second derivatives.
        all_axes = []
        for idx, key in enumerate(coeffs_full.keys(), start=1):
            ax = fig.add_subplot(3, 3, idx, projection="3d")
            surf = ax.plot_surface(X_for_dZ, Y_for_dZ, dZ[coeffs_full[key]])  # noqa: F841: yeah, `surf` is not used.
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(key)
            all_axes.append(ax)
            ground_truth = ground_truth_functions[key](X_for_dZ, Y_for_dZ)
            max_l1_error = np.max(np.abs(dZ[coeffs_full[key]] - ground_truth))
            print(f"    max absolute l1 error {key} = {max_l1_error:0.3g}")
        title_start = "Denoised local" if (σ > 0 and denoise_steps > 0) else "Local"
        fig.suptitle(f"{title_start} quadratic surrogate fit, noise σ = {σ:0.3g}")

        # Refitted second derivatives
        ax1 = fig.add_subplot(3, 3, 7, projection="3d")
        surf = ax1.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdx2)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("d2f/dx2 (refitted)")
        all_axes.append(ax1)
        ground_truth = ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdx2))
        print(f"    max absolute l1 error dx2 (refitted) = {max_l1_error:0.3g}")

        ax2 = fig.add_subplot(3, 3, 8, projection="3d")
        surf = ax2.plot_surface(X_for_dZ2, Y_for_dZ2, d2cross)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("d2f/dxdy (refitted)")
        all_axes.append(ax2)
        ground_truth = ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2cross))
        print(f"    max absolute l1 error dxdy (refitted) = {max_l1_error:0.3g}")

        ax3 = fig.add_subplot(3, 3, 9, projection="3d")
        surf = ax3.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdy2)  # noqa: F841: yeah, `surf` is not used.
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_title("d2f/dy2 (refitted)")
        all_axes.append(ax3)
        ground_truth = ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdy2))
        print(f"    max absolute l1 error dy2 (refitted) = {max_l1_error:0.3g}")

        link_3d_subplot_cameras(fig, all_axes)

        # l1 errors
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        def determine_value_range(Zs):
            """Produce a matching, symmetric value range for a colorbar, given several fields Z."""
            vs = []
            for Z in Zs:
                L, U = np.min(Z), np.max(Z)
                v = max(abs(L), abs(U))
                vs.append(v)
            return max(vs)
        def plot_one(ax, X, Y, Z, *, title, refZs=None):
            if refZs is None:
                refZs = [Z]
            v = determine_value_range(refZs)
            theplot = ax.pcolormesh(X, Y, Z, vmin=-v, vmax=v, cmap="RdBu_r")
            fig.colorbar(theplot, ax=ax)
            ax.set_aspect("equal")
            ax.set_title(title)
        errors = {"f": dZ[coeffs_full["f"]] - ground_truth_functions["f"](X, Y),
                  "dx": dZ[coeffs_full["dx"]] - ground_truth_functions["dx"](X_for_dZ, Y_for_dZ),
                  "dy": dZ[coeffs_full["dy"]] - ground_truth_functions["dy"](X_for_dZ, Y_for_dZ),
                  "dx2_raw": dZ[coeffs_full["dx2"]] - ground_truth_functions["dx2"](X_for_dZ, Y_for_dZ),
                  "dxdy_raw": dZ[coeffs_full["dxdy"]] - ground_truth_functions["dxdy"](X_for_dZ, Y_for_dZ),
                  "dy2_raw": dZ[coeffs_full["dy2"]] - ground_truth_functions["dy2"](X_for_dZ, Y_for_dZ),
                  "dx2_refit": d2zdx2 - ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2),
                  "dxdy_refit": d2cross - ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2),
                  "dy2_refit": d2zdy2 - ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2)}
        plot_one(axs[0, 0], X, Y, errors["f"], title="f")
        # Visualize the stencil size and shape.
        # The visualization center point [-3 N, 3 N] is arbitrary, as long as the whole stencil fits.
        idxs = np.array([[resolution - 3 * N_int, 3 * N_int]]) + stencil
        axs[0, 0].scatter(X[idxs[:, 0], idxs[:, 1]], Y[idxs[:, 0], idxs[:, 1]], s=1.0**2, c="#00000020", marker="o")
        refZs_1st = [errors["dx"], errors["dy"]]
        refZs_2nd = [errors["dx2_raw"], errors["dxdy_raw"], errors["dy2_raw"],
                     errors["dx2_refit"], errors["dxdy_refit"], errors["dy2_refit"]]
        plot_one(axs[0, 1], X_for_dZ, Y_for_dZ, errors["dx"], refZs=refZs_1st, title="dx")
        plot_one(axs[0, 2], X_for_dZ, Y_for_dZ, errors["dy"], refZs=refZs_1st, title="dy")
        plot_one(axs[1, 0], X_for_dZ, Y_for_dZ, errors["dx2_raw"], refZs=refZs_2nd, title="dx2")
        plot_one(axs[1, 1], X_for_dZ, Y_for_dZ, errors["dxdy_raw"], refZs=refZs_2nd, title="dxdy")
        plot_one(axs[1, 2], X_for_dZ, Y_for_dZ, errors["dy2_raw"], refZs=refZs_2nd, title="dy2")
        plot_one(axs[2, 0], X_for_dZ2, Y_for_dZ2, errors["dx2_refit"], refZs=refZs_2nd, title="dx2 (refitted)")
        plot_one(axs[2, 1], X_for_dZ2, Y_for_dZ2, errors["dxdy_refit"], refZs=refZs_2nd, title="dxdy (refitted)")
        plot_one(axs[2, 2], X_for_dZ2, Y_for_dZ2, errors["dy2_refit"], refZs=refZs_2nd, title="dy2 (refitted)")
        fig.suptitle("l1 error (fitted - ground truth)")
    print(f"    Done in {tim.dt:0.6g}s.")

if __name__ == '__main__':
    main()
    plt.show()
