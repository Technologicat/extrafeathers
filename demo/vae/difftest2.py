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
    batch_size = 8192

    # The input image is square, of size `resolution × resolution` pixels.
    resolution = 64

    xx = np.linspace(0, 1, resolution)
    yy = xx
    X, Y = np.meshgrid(xx, yy)
    X = tf.cast(X, dtype=tf.float32)
    Y = tf.cast(Y, dtype=tf.float32)

    # `prepare` only needs shape and dtype
    Z = tf.zeros([resolution, resolution], dtype=tf.float32)

    print("Setup...")
    preps, stencil = differentiate.prepare(N, X, Y, Z, p=p,
                                           format="LUp",
                                           low_vram=True, low_vram_batch_size=batch_size,
                                           print_statistics=True, indent=" " * 4)

    print("Spatial jacobian and hessian via WLSQM (with gradient tape enabled)...")
    a = tf.Variable(1.0, dtype=tf.float32, name="a", trainable=True)  # simple scalar parameter for demonstration
    with timer() as tim:
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

            # `dZ` = *spatial* jacobian and hessian of pixel image, via WLSQM: [f, dx, dy, dx2, dxdy, dy2]
            Z = tf.cast(Z, dtype=tf.float32)  # convert before call, to compile `solve_lu` for float32
            dZ = differentiate.solve_lu(*preps, Z, low_vram=True, low_vram_batch_size=batch_size)

            # out = dZ[0]  # or whatever you need to keep from `dZ`, for example...
            out = (dZ[3] + dZ[5])**2  # ...the spatial laplacian of the data, squared
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
        # NOTE: For the polynomial example, the result should be approximately 8:
        #   d/da[@a=1.0] [∇²(a X² + Y)]² = d/da[@a=1.0] [2 a]² = d/da[@a=1.0] [4 a²] = [8 a][@a=1.0] = 8
        doutda = tape.gradient(out, a) / resolution**2  # ∑_pixels [d(out)/da] / n_pixels
        print(f"    Average per-pixel value is {doutda:0.6g}")
    print(f"    Done in {tim.dt:0.6g}s.")

    print("Convert results to NumPy...")
    with timer() as tim:
        doutda = doutda.numpy()
    print(f"    Done in {tim.dt:0.6g}s.")
    # print(ddZda * (np.abs(ddZda) > 1e-4))


if __name__ == '__main__':
    main()
