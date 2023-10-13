"""Example of autodiff through a derivative field of a pixel image computed using WLSQM.

WLSQM produces the *spatial* jacobian and hessian of a pixel image, as well as the least-squares
fitted function values. The output is `[f, dx, dy, dx2, dxdy, dy2]`.

Autodiff can then produce the jacobian *of the WLSQM results* w.r.t. the *parameters*:
`d/da [f, dx, dy, dx2, dxdy, dy2]`.

This allows computing things like the derivative of a PDE residual w.r.t. neural network parameters,
allowing optimization against a physics-based loss.

For example, to use WLSQM plus a neural network as an AI solver for the 2D Poisson equation
in the unit square::

    d²z/dx² + d²z/dy² = f,  Ω = (0, 1)²

we can let a neural network produce candidate pixel images `z`. In the loss function, we then include
the physics loss term `λ ∑_pixels [dx2 + dy2 - load]²`, where `λ > 0` is a penalty parameter.

In the training process, autodiff against this loss computes `d/da [λ ∑_pixels [dx2 + dy2 - load]²]`,
which allows the optimizer to find neural network coefficients `a` that produce only pixel images that
fulfill the Poisson equation (in the mean squared error sense). Using various different load functions
in the training data then allows the neural network to learn to approximate the Laplace operator.

(Of course, to solve a boundary value problem, one must also enforce boundary conditions;
we gloss over this detail here. Note that WLSQM isn't very accurate near the edges anyway.)

There are other interesting applications, too. For example, we can specialize a VAE generative model
to produce only images that fulfill a given PDE. This *should* yield better AI outputs with less
training data, due to the physics-informed loss function - we are feeding in /a priori/ information
about the physics that guides the model toward generating the kind of outputs we want.

However, both those applications are beyond the scope here; this script is only a very basic
demonstration of the autodifferentiability of the WLSQM solver.

To run, issue the following command at the top-level directory of the project::

    TF_GPU_ALLOCATOR=cuda_malloc_async python -m demo.vae.difftest2
"""

from unpythonic import timer

import numpy as np
import tensorflow as tf

from . import differentiate
# from . import smoothing


def main():
    # WLSQM stencil parameters. See `difftest.py`.
    N, p = 2.5, 2.0

    # Batch size is *pixels* per batch.
    #
    # NOTE: If you want the per-pixel `tape.jacobian` (not just `tape.gradient` that sums over the pixels),
    # then autodiff is *very* VRAM hungry. On a 6 GB card, the largest that I could get to work was
    # `resolution = 64` with `batch_size = 32`.
    #
    # The forward pass doesn't use that much memory, and neither does the sum-over-pixels `tape.gradient`
    # (which is what one would use in a loss function anyway).
    #
    batch_size = 2048  # 8192

    # The input image is square, of size `resolution × resolution` pixels.
    resolution = 128

    xx = np.linspace(0, 1, resolution)
    yy = xx
    X, Y = np.meshgrid(xx, yy)

    print("Setup...")
    # For improved accuracy, we can `prepare` using `float64` coordinate data regardless of the solution `dtype` chosen here.
    solution_dtype = tf.float64
    preps, stencil = differentiate.prepare(N, X, Y, p=p,
                                           dtype=solution_dtype,  # compute and storage of the solution
                                           format="LUp",
                                           low_vram=True, low_vram_batch_size=batch_size,
                                           print_statistics=True, indent=" " * 4)
    # solve = differentiate.solve  # format="A"
    solve = differentiate.solve_lu  # format="LUp"

    # DEBUG
    def print_stencil(stencil):  # [[y0, x0], ...]
        miny = np.min(stencil[:, 0])
        maxy = np.max(stencil[:, 0])
        minx = np.min(stencil[:, 1])
        maxx = np.max(stencil[:, 1])
        stencil += np.array([-miny, -minx])
        mask = np.zeros([maxy - miny + 1, maxx - minx + 1], dtype=int)
        for y, x in stencil:
            mask[y, x] = 1
        for row in mask:
            for element in row:
                print("X" if element else " ", end="")
            print()
    print(f"Stencil shape (N = {N:0.3g}, p = {p:0.3g}):")
    print_stencil(stencil)

    print("Spatial jacobian and hessian via WLSQM (with gradient tape enabled)...")
    # a = tf.Variable(1.0, dtype=solution_dtype, name="a", trainable=True)  # simple scalar parameter for demonstration
    a = tf.Variable(tf.ones(tf.shape(X), dtype=solution_dtype), name="a", trainable=True)
    def fit_surrogate(Z):
        # `dZ` = *spatial* jacobian and hessian of pixel image, via WLSQM: [f, dx, dy, dx2, dxdy, dy2]
        dZ = solve(*preps, Z, low_vram=True, low_vram_batch_size=batch_size)
        if Z.dtype is tf.float64:
            return dZ  # fitted f, original 1st and 2nd derivatives
        # Refitting helps output accuracy at float32.
        ddx = solve(*preps, dZ[1], low_vram=True, low_vram_batch_size=batch_size)
        ddy = solve(*preps, dZ[2], low_vram=True, low_vram_batch_size=batch_size)
        # return tf.stack([dZ[0], dZ[1], dZ[2], ddx[1], 0.5 * (ddx[2] + ddy[1]), ddy[2]])  # fitted f, original 1st derivatives, refitted 2nd derivatives
        return tf.stack([dZ[0], ddx[0], ddy[0], ddx[1], 0.5 * (ddx[2] + ddy[1]), ddy[2]])  # fitted f, refitted 1st and 2nd derivatives

    with timer() as tim:
        # Here we must cast the coordinate data to the solution dtype,
        # because we will use them in the expression for `Z`.
        X = tf.cast(X, dtype=solution_dtype)
        Y = tf.cast(Y, dtype=solution_dtype)
        with tf.GradientTape() as tape:
            # NOTE: We must set up the quantity of interest inside the gradient tape extent,
            # so that the tape records all relevant computations. This is done rather naturally
            # in a neural network implementation (where the loss function is called inside the
            # gradient tape extent), but here we must do so explicitly.
            #
            # Thus this is where we set up the actual `Z` data (which depends on the demo parameter `a`).
            #
            # Z = tf.math.sin(a * X) * tf.math.cos(Y)
            Z = a * X**2 + Y

            # TODO: In a real NN application, we would here concatenate the pixels from several images
            # (from the same minibatch) and their `preps` appropriately. Basically, concatenate all of
            # the tensors on axis 0 (the batch axis), except leave `stencils` and `neighbors` alone.
            # See `prepare`.
            #
            # Another, maybe easier, way would be to WLSQM one minibatch member at a time, and gather
            # the results (e.g. the laplacian here) into a `tf.TensorArray`.
            #
            # For the first approach, we need to slightly modify the `solve` routines; they need to accept
            # flattened `z`, too. Then they become agnostic as to which minibatch member each pixel comes from.
            # Concatenating the `point_to_stencil` tensors sets up the correct stencils (since all images in
            # a minibatch are the same size, we can reuse the same stencils for each).
            dZ = fit_surrogate(Z)

            # out = dZ[0]  # or whatever you need to keep from `dZ`, for example...
            out = (dZ[3] + dZ[5])**2  # ...the spatial laplacian of the data, squared
            # out = smoothing.smooth_2d(int(np.ceil(N)), out, padding="VALID")
    print(f"    Done in {tim.dt:0.6g}s.")

    watched_variable_names = [var.name for var in tape.watched_variables()]
    print(f"    Gradient tape tracked gradients for variables: {watched_variable_names}")

    print("Autodiff w.r.t. parameters...")
    with timer() as tim:
        # For a scalar field `thing`, size [resolution, resolution]:
        #   `tape.jacobian(thing, a)`: d/da[thing] *pixelwise, for each pixel* -> field, [resolution, resolution]
        #   `tape.gradient(thing, a)`: ∑_pixels [d/da[thing]]                  -> scalar result
        # https://www.tensorflow.org/guide/autodiff
        # https://www.tensorflow.org/guide/advanced_autodiff#jacobians
        #
        # NOTE: For the polynomial example, the result should be 8, up to the accuracy of WLSQM:
        #   d/da[@a=1.0] [∇²(a X² + Y)]² = d/da[@a=1.0] [2 a]² = d/da[@a=1.0] [4 a²] = [8 a][@a=1.0] = 8
        doutda = tape.gradient(out, a)  # ∑_pixels [d(out)/da]
        # If we have just one scalar parameter `a`, then we should divide by the number of pixels.
        # But if we have one `a` for each pixel, then they're independent, and each one only affects its own pixel,
        # so summing over the pixels already gives the final result.
        # doutda /= resolution**2  # normalize result for scalar `a`
        doutda = doutda.numpy()
        print(f"    Average per-pixel value is {np.mean(doutda):0.6g}")
    print(f"    Done in {tim.dt:0.6g}s.")

    # print("Convert results to NumPy...")
    # with timer() as tim:
    #     doutda = doutda.numpy()
    # print(f"    Done in {tim.dt:0.6g}s.")
    # print(ddZda * (np.abs(ddZda) > 1e-4))

    import matplotlib.pyplot as plt
    # r = int(np.ceil(N))  # edge cut safety factor
    # dZ = dZ[:, r:-r, r:-r]  # not very accurate near the edges, so cut them away.
    idx_to_name = list(differentiate.coeffs_full.keys())  # list({v: k for k, v in differentiate.coeffs_full.items()}.values())
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for j in range(6):
        row, col = divmod(j, 3)
        ax = axs[row, col]
        theplot = ax.imshow(dZ[j].numpy(), origin="lower")
        fig.colorbar(theplot, ax=ax)
        ax.set_title(idx_to_name[j])
    # theplot = axs[2, 0].imshow(ddx[1].numpy(), origin="lower")
    # fig.colorbar(theplot, ax=axs[2, 0])
    # axs[2, 0].set_title("dx2 (refit)")
    # theplot = axs[2, 1].imshow(0.5 * (ddx[2] + ddy[1]).numpy(), origin="lower")
    # fig.colorbar(theplot, ax=axs[2, 1])
    # axs[2, 1].set_title("dxdy (refit)")
    # theplot = axs[2, 2].imshow(ddy[2].numpy(), origin="lower")
    # fig.colorbar(theplot, ax=axs[2, 2])
    # axs[2, 2].set_title("dy2 (refit)")

    plt.figure(2)
    # plt.imshow(doutda[r:-r, r:-r], origin="lower")
    plt.imshow(np.log10(np.abs(doutda)), origin="lower")
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()
