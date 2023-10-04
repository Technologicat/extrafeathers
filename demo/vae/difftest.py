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

from .differentiate import prepare, solve, coeffs_full, coeffs_diffonly

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
    # This is still horribly slow despite GPU acceleration. Performance is currently CPU-bound.
    # Denoising is the performance bottleneck. With numerically exact data, differentiation is acceptably fast.
    #
    # Even at 512 resolution, GPU utilization is under 20% (according to `nvtop`), and there is barely any
    # noticeable difference in the surrogate fitting speed. 512 is the largest that works, at least on
    # Quadro RTX 3000 mobile (RTX 2xxx based chip, 6 GB VRAM).
    #
    # At 768 or 1024, cuBLAS errors out (cuBlas call failed status = 14 [Op:MatrixSolve]).
    # Currently I don't know why - there should be no difference other than the batch size (the whole image
    # is sent in one batch). Solving a 6×6 linear system for 1024 RHSs should hardly take gigabytes of VRAM
    # even at float32.
    #
    # The hifiest algorithm (`prepare`/`solve`) *does* take gigabytes of VRAM, even at 256.
    resolution = 256

    # `N`: Neighborhood size parameter for surrogate fitting. The data used for fitting the local
    #      surrogate model for each pixel consists of a box of [2 * N + 1, 2 * N + 1] pixels,
    #      centered on the pixel being fitted. For pixels near edges and corners, the box is
    #      clipped to the data region. Furthermore, if the p-norm setting is other than "inf",
    #      the box is filtered to keep only points that are within a p-norm distance of `N`
    #      from the center.
    # `σ`: Optional, set to 0 to disable: stdev for per-pixel synthetic noise (i.i.d. gaussian).

    # When using Friedrichs smoothing, 8 is the largest numerically stable neighborhood size
    # (due to extrapolation at the edges).
    #
    # When smoothing only with least-squares, only VRAM is the limit.
    #
    # In any case, note the surrogate modeling assumption: a quadratic polynomial should be
    # able to reasonably describe the function locally in each neighborhood.
    #
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
    # At 256 resolution and 6 GB VRAM, about 530 points seems to be near the memory limit.
    # Euclidean neighborhoods with N = 13 seem the best choice.
    #
    # Note the count is always odd, because every other pixel has a symmetric pair,
    # but the center point doesn't.
    #
    # At N = 1, Euclidean neighborhoods would have 5 points, but the surrogate fitting
    # algorithm needs at least 7 to make the matrix invertible.
    #
    N, σ = 13, 0.001

    # # 2 seems enough for good results when the data is numerically exact.
    # N, σ = 2, 0.0

    # If σ > 0, how many times to loop the denoiser. Larger neighborhood sizes need less denoising.
    # If σ = 0, denoising is skipped, and this setting has no effect.
    #
    # To add synthetic noise, but skip denoising (to see how the results deteriorate),
    # set σ > 0 and `denoise_steps = 0`.
    denoise_steps = 2

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
    # n_neighborhood_points = (2 * N + 1)**2  # box
    n_neighborhood_points = len([[iy, ix] for iy in range(-N, N + 1)
                                          for ix in range(-N, N + 1)
                                          if (iy**2 + ix**2) <= N**2])  # circle
    print(f"    Neighborhood radius: {N} grid units ({n_neighborhood_points} grid points)")
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

    preps = prepare(N, X, Y, Z)  # Z is only needed here for shape and dtype.

    def denoise(N, X, Y, Z, *, indent=4):
        # Applying denoising in a loop allows removing larger amounts of noise.
        # Effectively, the neighboring patches communicate between iterations.
        for _ in range(denoise_steps):
            print(f"{indent * ' '}Denoising: step {_ + 1} of {denoise_steps}...")
            # tmp = hifier_differentiate(N, X, Y, Z, kernel=fit_quadratic)
            tmp = solve(*preps, Z)  # lsq
            Z = tmp[coeffs_full["f"]]
            # Z = smooth_2d(N, Z, padding="SAME")  # Friedrichs (eliminates high-frequency noise, but unstable extrapolation at edges/corners)
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
            tmp = solve(*preps, Z)
            noise_estimate = Z - tmp[coeffs_full["f"], :]
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
        dzdx = dZ[coeffs_diffonly["dx"], :]
        dzdy = dZ[coeffs_diffonly["dy"], :]
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
            ddzdx = solve(*preps, dzdx)[1:, :]
            print("        dy...")
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
    print(f"    Total wall time {tim_total.dt:0.6g}s.")

    # --------------------------------------------------------------------------------
    # Plot the results

    print("Plotting.")
    with timer() as tim:
        # Refitted second derivatives
        fig = plt.figure(2)
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        surf = ax1.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdx2)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("d2f/dx2")
        ground_truth = ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdx2))
        print(f"    max absolute l1 error dx2 (refitted) = {max_l1_error:0.3g}")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        surf = ax2.plot_surface(X_for_dZ2, Y_for_dZ2, d2cross)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("d2f/dxdy")
        ground_truth = ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2cross))
        print(f"    max absolute l1 error dxdy (refitted) = {max_l1_error:0.3g}")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        surf = ax3.plot_surface(X_for_dZ2, Y_for_dZ2, d2zdy2)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_title("d2f/dy2")
        ground_truth = ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2)
        max_l1_error = np.max(np.abs(ground_truth - d2zdy2))
        print(f"    max absolute l1 error dy2 (refitted) = {max_l1_error:0.3g}")

        fig.suptitle(f"Local quadratic surrogate fit, refitted second derivatives, noise σ = {σ:0.3g}")
        link_3d_subplot_cameras(fig, [ax1, ax2, ax3])

        # Function and the raw first and second derivatives
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
        title_start = "Denoised local" if (σ > 0 and denoise_steps > 0) else "Local"
        fig.suptitle(f"{title_start} quadratic surrogate fit, noise σ = {σ:0.3g}")
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
        plot_one(axs[2, 0], X_for_dZ2, Y_for_dZ2, d2zdx2 - ground_truth_functions["dx2"](X_for_dZ2, Y_for_dZ2), "dx2 (refitted)")
        plot_one(axs[2, 1], X_for_dZ2, Y_for_dZ2, d2cross - ground_truth_functions["dxdy"](X_for_dZ2, Y_for_dZ2), "dxdy (refitted)")
        plot_one(axs[2, 2], X_for_dZ2, Y_for_dZ2, d2zdy2 - ground_truth_functions["dy2"](X_for_dZ2, Y_for_dZ2), "dy2 (refitted)")
        fig.suptitle("l1 error (fitted - ground truth)")
    print(f"    Done in {tim.dt:0.6g}s.")

if __name__ == '__main__':
    main()
    plt.show()
