#!/usr/bin/env python
"""Use the weighted least squares meshfree method to fit a local quadratic surrogate to compute derivatives.

This method produces approximations for both the jacobian and the hessian in one go, by solving local 5×5 equation systems.

This is basically a technique demo; this can be GPU-accelerated with TF, so we can use it to evaluate the
spatial derivatives in the PDE residual in a physically informed loss for up to 2nd order PDEs.

The potential issue is that the VAE output might contain some noise, so we need a method that can handle noisy input.

We need only the very basics here. A more complete Cython implementation, and documentation:
    https://github.com/Technologicat/python-wlsqm
"""

import numpy as np
import sympy as sy
import tensorflow as tf

import matplotlib.pyplot as plt

def main():
    # --------------------------------------------------------------------------------
    # Parameters
    N = 3  # neighborhood size parameter for surrogate fitting
    σ = 0.01  # optional: simulate i.i.d. gaussian noise in data
    xx = np.linspace(0, np.pi, 21)
    yy = xx

    # --------------------------------------------------------------------------------
    # Function
    x, y = sy.symbols("x, y")
    f = sy.sin(x) * sy.cos(y)
    # f = x**2 + y

    # --------------------------------------------------------------------------------
    # General init

    # Differentiate symbolically to obtain ground truths for the jacobian and hessian to benchmark against.
    dfdx = sy.diff(f, x, 1)
    dfdy = sy.diff(f, y, 1)
    d2fdx2 = sy.diff(f, x, 2)
    d2fdxdy = sy.diff(sy.diff(f, x, 1), y, 1)
    d2fdy2 = sy.diff(f, y, 2)

    f = sy.lambdify((x, y), f)
    dfdx = sy.lambdify((x, y), dfdx)
    dfdy = sy.lambdify((x, y), dfdy)
    d2fdx2 = sy.lambdify((x, y), d2fdx2)
    d2fdxdy = sy.lambdify((x, y), d2fdxdy)
    d2fdy2 = sy.lambdify((x, y), d2fdy2)
    ground_truths = {"dx": dfdx, "dy": dfdy, "dx2": d2fdx2, "dxdy": d2fdxdy, "dy2": d2fdy2}

    X, Y = np.meshgrid(xx, yy)
    f = f(X, Y)

    neighbors = [[iy, ix] for iy in range(-N, N + 1)
                          for ix in range(-N, N + 1)
                          if not (iy == 0 and ix == 0)]
    neighbors = np.array(neighbors, dtype=int)
    print(f"Using {len(neighbors)}-point neighborhoods (N = {N}).")

    # --------------------------------------------------------------------------------
    # Noisy input simulation.
    if σ > 0:
        # Corrupt the data with synthetic noise
        noise = np.random.normal(loc=0.0, scale=σ, size=np.shape(X))
        f += noise

        # Smooth away the noise so we get a reasonable hessian, too.
        #
        # To smooth the data, we use a discrete convolution with the Friedrichs mollifier.
        # A continuous version is sometimes used as a differentiation technique:
        #
        #    Ian Knowles and Robert J. Renka. Methods of numerical differentiation of noisy data.
        #    Electronic journal of differential equations, Conference 21 (2014), pp. 235-246.
        #
        # but we use a simple discrete implementation, and only as a preprocessor. We send the output
        # to the quadratic surrogate fitter.
        def friedrichs_mollifier(x, *, eps=0.001):  # not normalized!
            return np.where(np.abs(x) < 1 - eps, np.exp(-1 / (1 - x**2)), 0.)
        offset_X, offset_Y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))  # neighbors
        offset_R = np.sqrt(offset_X**2 + offset_Y**2)
        rmax = np.ceil(np.max(offset_R))  # the offset distance at which we want the mollifier to vanish
        smoothing_kernel = friedrichs_mollifier(offset_R / rmax)
        smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)
        f = tf.expand_dims(f, axis=0)  # batch
        f = tf.expand_dims(f, axis=-1)  # channels
        smoothing_kernel = tf.expand_dims(smoothing_kernel, axis=-1)  # input channels
        smoothing_kernel = tf.expand_dims(smoothing_kernel, axis=-1)  # output channels
        f = tf.nn.convolution(f, smoothing_kernel, padding="VALID")
        f = tf.squeeze(f, axis=-1)  # channels
        f = tf.squeeze(f, axis=0)  # batch
        f = f.numpy()
        # The edges are nonsense with padding="SAME", so we use "VALID", and chop off the edges of X and Y correspondingly.
        X = X[N:-N, N:-N]
        Y = Y[N:-N, N:-N]

    # --------------------------------------------------------------------------------
    # # Fit a 2nd order surrogate polynomial to obtain derivatives.
    # # Old inefficient code.
    # TODO: how to vectorize this?
    # TODO: how to do this in TF? (EagerTensor does not support item assignment)
    # ny, nx = np.shape(X)
    # fcoeffs = lambda dx, dy: [dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]  # Taylor
    # ncoeffs = len(fcoeffs(0, 0))
    # coeff = {"dx": 0, "dy": 1, "dx2": 2, "dxdy": 3, "dy2": 4}
    # As = []
    # bs = []
    # for iy in range(N, ny - N):
    #     for ix in range(N, nx - N):
    #         A = np.zeros((ncoeffs, ncoeffs))
    #         b = np.zeros((ncoeffs,))
    #         for offset_y, offset_x in neighbors:
    #             dx = X[iy + offset_y, ix + offset_x] - X[iy, ix]
    #             dy = Y[iy + offset_y, ix + offset_x] - Y[iy, ix]
    #             c = fcoeffs(dx, dy)
    #             for j in range(ncoeffs):
    #                 for n in range(ncoeffs):
    #                     A[j, n] += c[j] * c[n]
    #                 b[j] += (f[iy + offset_y, ix + offset_x] - f[iy, ix]) * c[j]
    #         As.append(A)
    #         bs.append(b)
    # As = tf.stack(As, axis=0)
    # bs = tf.stack(bs, axis=0)
    # bs = tf.expand_dims(bs, axis=-1)
    # df = tf.linalg.solve(As, bs)
    # df = tf.squeeze(df, axis=-1)
    # df = tf.reshape(df, (ny - 2 * N, nx - 2 * N, ncoeffs))  # [iy, ix, c]

    # --------------------------------------------------------------------------------
    # Fit a 2nd order surrogate polynomial to obtain derivatives.

    # Since we have a uniform grid in this application, the distance matrix of neighbors for each point is the same,
    # so we need to assemble only one.
    #
    # Note A is 5×5 regardless of N, but for large N, assembly takes longer because there are more contributions
    # to each matrix element.
    #
    # Note we lose the edges, like in a convolution with padding="VALID", essentially for the same reason.
    # Here this is mathematically trivial to fix (just build A and b without the missing neighbors, or take
    # more neighbors from the existing side - even the sizes of A and b do not change), but the code becomes
    # unwieldy for a simple example.
    #
    # fcoeffs = lambda dx, dy: [dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]  # Taylor

    # Derivative scaling for numerical stability: x' := x / xscale  ⇒  d/dx → (1 / xscale) d/dx'.
    # Choose xscale so that the magnitudes are near 1. Similarly for y. We use the grid spacing as the scale.
    xscale = X[0, 1] - X[0, 0]
    yscale = Y[1, 0] - Y[0, 0]
    def fcoeffs(dx, dy):
        dx = dx / xscale
        dy = dy / yscale
        return [dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2]  # Taylor

    ncoeffs = len(fcoeffs(0, 0))
    coeff = {"dx": 0, "dy": 1, "dx2": 2, "dxdy": 3, "dy2": 4}
    A = np.zeros((ncoeffs, ncoeffs))
    iy, ix = N, N  # Any node in the interior is fine, since the local topology and geometry are the same for all of them.
    for offset_y, offset_x in neighbors:
        dx = X[iy + offset_y, ix + offset_x] - X[iy, ix]
        dy = Y[iy + offset_y, ix + offset_x] - Y[iy, ix]
        c = fcoeffs(dx, dy)
        for j in range(ncoeffs):
            for n in range(ncoeffs):
                A[j, n] += c[j] * c[n]
    # Form the right-hand side for each point. This is the only part that depends on the data values f.
    ny, nx = np.shape(X)
    bs = []
    for iy in range(N, ny - N):
        for ix in range(N, nx - N):
            b = np.zeros((ncoeffs,))
            for offset_y, offset_x in neighbors:
                dx = X[iy + offset_y, ix + offset_x] - X[iy, ix]
                dy = Y[iy + offset_y, ix + offset_x] - Y[iy, ix]
                c = fcoeffs(dx, dy)
                for j in range(ncoeffs):
                    b[j] += (f[iy + offset_y, ix + offset_x] - f[iy, ix]) * c[j]
            bs.append(b)
    bs = tf.stack(bs, axis=1)

    # The solution of the linear systems (one per data point) yields the jacobian and hessian of the surrogate.
    df = tf.linalg.solve(A, bs)  # [ncoeffs, n_datapoints]

    # Undo the derivative scaling,  d/dx' → d/dx
    scale = tf.constant([xscale, yscale, xscale**2, xscale * yscale, yscale**2])
    scale = tf.expand_dims(scale, axis=-1)  # for broadcasting to all data points
    df = df / scale

    df = tf.reshape(df, (ncoeffs, ny - 2 * N, nx - 2 * N))

    # --------------------------------------------------------------------------------
    # Plot the results

    # https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
    fig = plt.figure(1)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    surf = ax.plot_surface(X, Y, f, linewidth=0, antialiased=False)  # noqa: F841
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("f")
    all_axes = [ax]
    for idx, key in enumerate(coeff.keys(), start=2):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        surf = ax.plot_surface(X[N:-N, N:-N], Y[N:-N, N:-N], df[coeff[key], :, :], linewidth=0, antialiased=False)  # noqa: F841
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(key)
        all_axes.append(ax)

        ground_truth = ground_truths[key](X[N:-N, N:-N], Y[N:-N, N:-N])
        max_l1_error = np.max(np.abs(ground_truth - df[coeff[key], :, :]))
        print(f"max absolute l1 error {key} = {max_l1_error:0.3g}")
    fig.suptitle(f"Local quadratic surrogate fit, noise σ = {σ:0.3g}")

    # Link the subplot cameras (view angles and zooms)
    # https://github.com/matplotlib/matplotlib/issues/11181
    def on_move(event):
        sender_ax = [ax for ax in all_axes if event.inaxes == ax]
        if not sender_ax:
            return
        assert len(sender_ax) == 1
        sender_ax = sender_ax[0]
        other_axes = [ax for ax in all_axes if ax is not sender_ax]
        if sender_ax.button_pressed in sender_ax._rotate_btn:
            for ax in other_axes:
                ax.view_init(elev=sender_ax.elev, azim=sender_ax.azim)
        elif sender_ax.button_pressed in sender_ax._zoom_btn:
            for ax in other_axes:
                ax.set_xlim3d(sender_ax.get_xlim3d())
                ax.set_ylim3d(sender_ax.get_ylim3d())
                ax.set_zlim3d(sender_ax.get_zlim3d())
        else:
            return
        fig.canvas.draw_idle()
    on_move_event = fig.canvas.mpl_connect('motion_notify_event', on_move)  # noqa: F841

if __name__ == '__main__':
    main()
    plt.show()
