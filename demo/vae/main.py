#!/usr/bin/python
"""Convolutional variational autoencoder (CVAE) implemented in Keras and TensorFlow.

This is a simple technology demo, condensed and improved from existing online VAE tutorials,
that just approximates MNIST digits. The goal of introducing a VAE in the context of
`extrafeathers` is to eventually make a PINN (physically informed neural network) for
the acceleration of numerical simulations in digital twin applications.

The idea is that a PDE solution field, projected onto a uniform grid, can be thought of as
a multichannel image (like an RGB image, but tensor components instead of colors). A PINN
is produced by adding an extra term to the loss function, penalizing the residual of the PDE,
as evaluated using the reconstructed output. The encoder part of the VAE then acts as a
lossy data compressor, reducing the solution field into a low-dimensional representation
(which can be decompressed using the decoder part of the VAE).

For the acceleration of simulations, a VAE is a particularly useful type of neural network,
because its latent representation is continuous. Given a trained VAE, we can train another
neural network to act as a time evolution operator in the low-dimensional latent space, thus
yielding a ROM (reduced order model) for real-time simulation.

Note that a VAE is essentially an interpolator, so it only helps with deployed models, not in
research into new situations. For the output to be meaningful, the situation being simulated,
or a reasonable approximation thereof, must exist in the training data of the VAE. This
limitation is not too different from classical ROM techniques such as proper orthogonal
decomposition (POD); that too is an interpolator, which can only reproduce behavior that
exists in the input data.

As development proceeds, the next planned step is to implement a CVAE to approximate the
solution fields of the classical vortex shedding benchmark (flow over cylinder).

This script operates in two modes:
   1) As the main program, to train the model.
   2) As an importable module, to use the trained model.

To train the model, open a terminal in the top-level `extrafeathers` directory, and:

  python -m demo.vae.main

The model will be saved in `saved_model_dir`, defined in config below.

To load the trained model, in an IPython session:

  import demo.vae.main as main
  main.model.my_load(main.saved_model_dir)

Now you can e.g.:

  import matplotlib.pyplot as plt
  main.plot_and_save_latent_image(10)
  plt.show()

The implementation is based on combining material from these two tutorials:
  https://www.tensorflow.org/tutorials/generative/cvae
  https://keras.io/examples/generative/vae/

References:

  Diederik P. Kingma and Max Welling. 2019. An introduction to variational autoencoders.
  https://arxiv.org/abs/1906.02691

  Gabriel Loaiza-Ganem and John P. Cunningham. 2019. The continuous Bernoulli: fixing a pervasive error
  in variational autoencoders.
  https://arxiv.org/abs/1907.06845

  Shuyu Lin, Stephen Roberts, Niki Trigoni, and Ronald Clark. 2019. Balancing reconstruction quality
  and regularisation in evidence lower bound for variational autoencoders.
  https://arxiv.org/abs/1909.03765

  Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. MIT press.
  http://www.deeplearningbook.org
"""

# The REPL server allows inspecting/saving anything accessible from module-global scope
# while the process is live. To connect, `python -m unpythonic.net.client localhost`.
import unpythonic.net.server as repl_server

# TODO: add ELBO history progress plot (we should also compute statistics of these over several runs)
# TODO: use an early-stopping criterion?
# TODO: plot the original data locations in the latent space (like in the Keras tutorial)

# TODO: For use with PDE solution fields:
#   - Project the PDE solution to a uniform grid (uniform, since we use a convolutional NN)
#   - Change the decoder model to a Gaussian (with learnable variance), as suggested in the paper by Lin et al.
#   - Or better yet, compute the distribution of the actual fields (observed data!) and model that

import glob
import os
import pathlib
import typing

from unpythonic import ETAEstimator, timer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

# import PIL
import imageio

from extrafeathers import plotmagic

# --------------------------------------------------------------------------------
# Config

# batch_size = 32  # for CPU
# batch_size = 128  # faster on GPU, still acceptable generalization on discrete Bernoulli
batch_size = 64  # acceptable generalization on continuous Bernoulli

latent_dim = 2  # use a 2-dimensional latent space so that we can easily visualize the results

epochs = 100

# For a discussion of NN optimization methods, see the Deep Learning book by Goodfellow et al.
optimizer = tf.keras.optimizers.Adam(1e-4)

# Saving
output_dir = "demo/output/vae/"
saved_model_dir = "demo/output/vae/my_model"  # for trained model reloading later

fig_basename = "epoch"  # -> "epoch_0000.png" and so on; must be a unique prefix among output filenames
fig_format = "png"
latent_vis_basename = "latent_space"
anim_filename = "cvae.gif"

# --------------------------------------------------------------------------------
# Helper for deleting previously saved model (to save new one cleanly)

def _delete_directory_recursively(path: str) -> None:
    """Delete a directory recursively, like 'rm -rf' in the shell.

    Ignores `FileNotFoundError`, but other errors raise. If an error occurs,
    some files and directories may already have been deleted.
    """
    path = pathlib.Path(path).expanduser().resolve()

    for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
        for x in files:
            try:
                os.unlink(os.path.join(root, x))
            except FileNotFoundError:
                pass

        for x in dirs:
            try:
                os.rmdir(os.path.join(root, x))
            except FileNotFoundError:
                pass

    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass

def _create_directory(path: str) -> None:
    p = pathlib.Path(path).expanduser().resolve()
    pathlib.Path.mkdir(p, parents=True, exist_ok=True)

def _clear_and_create_directory(path: str) -> None:
    _delete_directory_recursively(path)
    _create_directory(path)

# --------------------------------------------------------------------------------
# NN architecture

# Encoder/decoder architecture modified from https://keras.io/examples/generative/vae/
#
# Encoder differences:
#   - No z value in output, we use the reparameterize function instead.
#     The custom Sampling layer in the original is a neat technique,
#     but this is conceptually cleaner (encoder output is explicitly
#     a distribution (represented as parameters), not a sample from it).
#
# Decoder differences:
#   - We handle the final (sigmoid) activation of the decoder manually later.
#   - We have added an extra Dense layer of 16 units at the input side
#     to more closely mirror the structure of the encoder.
#   - The architecture is an exact mirror image of the encoder.
def make_encoder():
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                               strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                               strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    # VRAM saving trick from the Keras example: the encoder has *two* outputs: mean and logvar. Hence,
    # if we add a small dense layer after the convolutions, and connect that to the outputs, we will have
    # much fewer trainable parameters (in total) than if we connect the last convolution layer to the outputs
    # directly. As pointed out by:
    # https://linux-blog.anracom.com/2022/10/23/variational-autoencoder-with-tensorflow-2-8-xii-save-some-vram-by-an-extra-dense-layer-in-the-encoder/
    x = tf.keras.layers.Dense(units=16, activation="relu")(x)
    # No activation function in the output layers - we want arbitrary real numbers as output.
    # The outputs will be interpreted as `(μ, log σ)` for the variational posterior qϕ(z|x).
    # A uniform distribution for these quantities (from the random initialization of the NN)
    # is a good prior for unknown location and scale parameters, see e.g.:
    #   https://en.wikipedia.org/wiki/Principle_of_transformation_groups
    #   https://en.wikipedia.org/wiki/Principle_of_maximum_entropy
    z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var")(x)
    encoder_outputs = [z_mean, z_log_var]
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder

def make_decoder():
    # decoder - exact mirror image of encoder (w.r.t. tensor sizes at each step)
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    # Here we add the dense layer just for architectural symmetry with the encoder.
    x = tf.keras.layers.Dense(units=16, activation="relu")(decoder_inputs)
    x = tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu")(x)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
                                        strides=2, padding="same")(x)
    # No activation function in the output layer - we want arbitrary real numbers as output.
    # The output will be interpreted as parameters `P` for the observation model pθ(x|z).
    # Here we want just something convenient that we can remap as necessary.
    decoder_outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(x)
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder

# TODO: `CVAE` does not conform to the standard Keras Model API, since it does not implement `call`,
# TODO: and although we have a custom `train_step`, it's a separate function, not a method of `CVAE`.
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = make_encoder()
        self.decoder = make_decoder()

    @tf.function
    def sample(self, z=None):
        """Sample from the observation model pθ(x|z), returning the pixel-wise means.

        `z`: tf array of size `(batch_size, latent_dim)`.
             If not specified, a batch of 100 random samples is returned.
        """
        if z is None:
            # Sample the code points from the latent prior.
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)

    def encode(self, x):
        """x → parameters of variational posterior qϕ(z|x), namely `(μ, log σ)`."""
        # # If we had a single output layer of double the size, we could do it like this:
        # mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        # return mean, logvar
        # But having two outputs explicitly, we can just:
        return self.encoder(x)

    def reparameterize(self, mean, logvar):
        """Map  (μ, log σ) → z  stochastically.

        Return (ε, z) (the same ε sample is needed in the ELBO computation).
        """
        # https://datascience.stackexchange.com/questions/51086/valueerror-cannot-convert-a-partially-known-tensorshape-to-a-tensor-256
        # `tf.shape` is dynamic; `mean.shape` is static (see docstring of `tf.shape`) and
        # does not work if one of the dimensions is not yet known (e.g. the batch size).
        # This is important to make the model saveable, because `Model.save` calls the
        # `call` method with an unspecified batch size (to build the computational graph).
        # eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps, eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """z → parameters of observation model pθ(x|z)"""
        P = self.decoder(z)
        if apply_sigmoid:
            # For the discrete Bernoulli distribution, the mean is the same as the Bernoulli parameter,
            # which is the same as the probability of the output taking on the value 1 (instead of 0).
            p = tf.sigmoid(P)
            return p
        return P

    # TODO: Saving a CVAE instance using the official Keras serialization API doesn't work yet.
    # TODO: So for now, we separately save the encoder and decoder models (both are Functional
    # TODO: models that support saving natively).
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    def my_save(self, path=saved_model_dir):
        _clear_and_create_directory(path)
        p = pathlib.Path(path).expanduser().resolve()
        self.encoder.save(str(p / "encoder"))
        self.decoder.save(str(p / "decoder"))
    def my_load(self, path=saved_model_dir):
        p = pathlib.Path(path).expanduser().resolve()
        self.encoder = tf.keras.models.load_model(str(p / "encoder"))
        self.decoder = tf.keras.models.load_model(str(p / "decoder"))

    # def get_config(self):
    #     return {"latent_dim": self.latent_dim,
    #             "encoder": self.encoder,
    #             "decoder": self.decoder}
    # @classmethod
    # def from_config(cls, config):
    #     model = cls(config["latent_dim"])
    #     model.encoder = config["encoder"]
    #     model.decoder = config["decoder"]
    #     return model
    # # to make a custom object saveable, it must have a call method
    # def call(self, inputs):
    #     mean, logvar = self.encode(inputs)
    #     ignored_eps, z = self.reparameterize(mean, logvar)
    #     xhat = self.decode(z, apply_sigmoid=True)
    #     return xhat

model = CVAE(latent_dim)

# --------------------------------------------------------------------------------
# Loss function

# Note that since we have defined the reparameterization as
#   z = mean + eps * exp(logvar / 2)
# inverting yields
#   eps = (z - mean) * exp(-logvar / 2)
# and
#   eps² = (z - mean)² * exp(-logvar)
# so calling log_normal_pdf(z, mean, logvar) actually yields
#   sum_i(-0.5 * eps_i**2 + logvar + log2pi)
# which matches Kingma and Welling (2019, algorithm 2).
def log_normal_pdf(x, mean, logvar, raxis=1):
    log2pi = tf.math.log(2 * np.pi)
    return tf.reduce_sum(-0.5 * ((x - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
                         axis=raxis)

def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    """Compute the log normalizing constant of a continuous Bernoulli distribution.

    Returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and
    a Taylor approximation in [l_lim, u_lim] (there is a singularity in the
    general formula at lam=0.5).

    Numerically stable formulation.

    Taken from the GitHub repo for the paper by Loaiza-Ganem and Cunningham (2019):
        https://github.com/cunningham-lab/cb_and_cc
    """
    # cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5
    # as tf.where evaluates both options, regardless of the value of the condition.
    cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tf.math.log(tf.abs(2.0 * tf.atanh(1 - 2.0 * cut_lam))) - tf.math.log(tf.abs(1 - 2.0 * cut_lam))
    taylor = tf.math.log(2.0) + 4.0 / 3.0 * tf.pow(lam - 0.5, 2) + 104.0 / 45.0 * tf.pow(lam - 0.5, 4)
    return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), log_norm, taylor)

def compute_loss(model, x):
    """VAE loss function: negative ELBO.

    Evaluated by drawing a single-sample Monte Carlo estimate. Kingma and Welling (2019) note
    that this yields an unbiased estimator for the expectation that appears in the ELBO formula.

    Note that we can't use any expensive methods here, because this runs in the inner loop of the
    NN optimization.
    """
    # Without explanation this calculation is overly magic, so let's comment step by step.

    # Encode the input data (pixels) into parameters for the variational posterior qϕ(z|x):
    #   x → (μ(ϕ, x), log σ(ϕ, x))
    # Here ϕ are the encoder NN coefficients: NN_enc = NN_enc(ϕ, x).
    mean, logvar = model.encode(x)

    # Draw a single sample z ~ qϕ(z|x), using a single noise sample ε ~ p(ε) and the deterministic
    # reparameterization transformation z = g(ε, ϕ, x). In the implementation, actually z = g(μ, log σ);
    # the dependencies on ϕ and x have been absorbed into μ(ϕ, x) and log σ(ϕ, x). The `reparameterize`
    # function internally draws the noise sample ε, so we don't need to supply it here.
    #
    # We choose our class of variational posteriors (which we optimize over) as factorized Gaussian,
    # mainly for convenience. Thus we interpret the (essentially arbitrary) numbers coming from the
    # encoder as (μ, log σ) for a factorized Gaussian, and plug them in in those roles.
    #
    # Note z is encoded stochastically; even feeding in the same x produces a different z each time
    # (since z is sampled from the variational posterior).
    eps, z = model.reparameterize(mean, logvar)

    # Decode the sampled `z`, obtain parameters (at each pixel) for observation model pθ(x|z).
    # Here θ are the decoder NN coefficients: NN_dec = NN_dec(θ, z).
    #
    # We implement the classic VAE: we choose our class of observation models as factorized Bernoulli
    # (pixel-wise). Thus we interpret the (essentially arbitrary) numbers coming from the decoder as
    # logits for a factorized Bernoulli distribution, and plug them in in that role.
    #
    # The original tutorial implementations are unnecessarily confusing here; logits and the binary
    # cross-entropy are a sideshow, specific to the discrete Bernoulli observation model of the classic
    # VAE, which doesn't even make sense for continuous (grayscale) data (hence the unnecessary
    # binarization of the input data, which hurts quality).
    #
    # How to compute pθ(x|z) in general, for any VAE: we just take the observation parameters P computed by
    # P = NN_dec(θ, z) at the sampled z (thus accounting for the dependence on z), and evaluate the known
    # function pθ(x|z) (parameterized by P and x) with those parameters P at the input x.
    #
    # Note that in a VAE, unlike a classical AE, the decoded output is not directly x-hat (i.e. an
    # approximation of the input data point), but parameters for a distribution that can be used to
    # compute x-hat. Strictly, in a VAE, x-hat is a (pixelwise) distribution. Many implementations
    # return the mean of that distribution as the decoded image.
    #
    # For more details, see the VAE tutorial paper by Kingma and Welling (2019, algorithm 2).
    P = model.decode(z)

    # Evaluate a single-sample Monte Carlo (MC) estimate of the ELBO.
    #
    # We basically need the logarithms of the three probabilities pθ(x|z) (observation model),
    # pθ(z) (latent prior), and qϕ(z|x) (variational posterior), evaluated for the current model (θ, ϕ),
    # at the input x and the corresponding encoded z. The encoded z is sampled from qϕ(z|x), which is
    # available for the current iterate of the variational parameters ϕ.
    #
    # IMPORTANT for consistency: We use the *input* x and the *sampled* z in all terms of the MC estimate.
    # This is because we are evaluating the ELBO at the input x.
    #
    # For example, evaluating the latent prior at the sampled z gives the probability that the prior pθ(z)
    # assigns to the z drawn from the approximate posterior qϕ(z|x).
    #
    # In the observation term pθ(x|z), our z sample (which the observation is conditioned on) is drawn from
    # qϕ(z|x), which in turn is conditioned on the input x.
    #
    # # The original tutorial implementations did it like this:
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=P, labels=x)
    # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # # Discrete Bernoulli, computing pθ(x|z) ourselves explicitly:
    # p = tf.sigmoid(P)  # interpret decoder output as logits; map into probabilities
    # p = tf.clip_by_value(p, 1e-4, 1 - 1e-4)  # avoid log(0)
    # logpx_z = tf.reduce_sum(x * tf.math.log(p) + (1 - x) * tf.math.log(1 - p),
    #                         axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # Continuous Bernoulli - add a log-normalizing constant to make the probability distribution sum to 1.
    # Note the output is now the parameter λ of the continuous Bernoulli distribution, not directly a
    # probability; and the mean of the continuous Bernoulli distribution is also different from λ.
    # But still, in practice λ works well as-is as a deterministic output value for the decoder.
    #
    # As for how to apply the normalization constant, see the original implementation by Loaiza-Ganem
    # and Cunningham: https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/cb_vae_mnist.ipynb
    lam = tf.sigmoid(P)  # interpret decoder output as logits; map into λ parameter of continuous Bernoulli
    lam = tf.clip_by_value(lam, 1e-4, 1 - 1e-4)  # avoid log(0)
    logpx_z = tf.reduce_sum(x * tf.math.log(lam) + (1 - x) * tf.math.log(1 - lam) + cont_bern_log_norm(lam),
                            axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # We choose the latent prior pθ(z) to be a spherical unit Gaussian, N(0, 1).
    # Note this spherical unit Gaussian is distinct from the one we used for the noise variable ε.
    # Note also the function `log_normal_pdf` takes `log σ`, not bare `σ`.
    logpz = log_normal_pdf(z, 0., 0.)                    # log pθ(z)   (latent prior, at *sampled* z)

    logqz_x = log_normal_pdf(z, mean, logvar)            # log qϕ(z|x) (variational posterior)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)    # ELBO (sign flipped → ELBO loss)

@tf.function
def train_step(model, x, optimizer):
    """Execute one training step, computing and applying gradients via backpropagation."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# --------------------------------------------------------------------------------
# Main program - training

def plot_and_save_epoch_image(model: CVAE, epoch: int, test_sample: tf.Tensor, figno: int = 1) -> None:
    """Plot image of test sample and the corresponding prediction (by feeding the sample through the CVAE)."""
    batch_size, n_pixels_y, n_pixels_x, n_channels = tf.shape(test_sample).numpy()
    assert batch_size == 16, f"This function currently assumes a test sample of size 16, got {batch_size}"
    assert n_channels == 1, f"This function currently assumes grayscale images, got {n_channels} channels"

    mean, logvar = model.encode(test_sample)
    ignored_eps, z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    n = 4  # how many images per row/column; sqrt(batch_size)
    image_width = (2 * n + 1) * n_pixels_x  # extra empty column at center, as separator
    image_height = n * n_pixels_y
    image = np.zeros((image_height, image_width))

    for i in range(batch_size):
        x_orig = test_sample[i, :, :, 0]
        x_hat = predictions[i, :, :, 0]
        row, base_col = divmod(i, n)
        col1 = base_col  # original image (input)
        col2 = base_col + n + 1  # reconstructed image
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col1 * n_pixels_x: (col1 + 1) * n_pixels_x] = x_orig.numpy()
        image[row * n_pixels_y: (row + 1) * n_pixels_y,
              col2 * n_pixels_x: (col2 + 1) * n_pixels_x] = x_hat.numpy()

    plt.figure(figno, figsize=(8, 4))
    plt.clf()
    plt.imshow(image, cmap="Greys_r")
    plt.axis("off")
    plt.title(f"Epoch {epoch} (left: input $\\mathbf{{x}}$, right: prediction $\\hat{{\\mathbf{{x}}}}$)")
    plt.tight_layout()
    _create_directory(output_dir)
    plt.savefig(f"{output_dir}{fig_basename}_{epoch:04d}.{fig_format}")
    plt.draw()
    plotmagic.pause(0.1)  # force redraw

def plot_and_save_latent_image(n: int = 20, model: typing.Optional[CVAE] = None, digit_size: int = 28,
                               grid: str = "quantile", eps: float = 3, figno: int = 1) -> None:
    """Plot n × n digit images decoded from the latent space.

    `digit_size`: width/height of each digit image (square-shaped), in pixels.
                  Must match what the model was trained for.

    `grid`: grid spacing type; one of "linear" or "quantile" (default)

            A quantile grid accounts for the shape of the spherical gaussian latent
            prior, placing more emphasis on the region near the origin, where most
            of the prior's probability mass is concentrated. This grid is linear
            in cumulative probability.

            A linear grid effectively emphasizes the faraway regions of the
            latent space, since the prior does not have much probability mass there.
            This grid is linear directly in the coordinates of the latent space.

            Grid min/max are always taken to be ±εσ in the latent space.

    `eps`:  ε for grid lower/upper limit ±εσ. E.g. the default 3 means ±3σ.

    `model`: `CVAE` instance, or `None` to use the default instance.
    `figno`: matplotlib figure number.
    """
    if model is None:
        def get_default_model():
            global model
            return model
        model = get_default_model()
    assert isinstance(model, CVAE)
    assert grid in ("linear", "quantile")

    gaussian = tfp.distributions.Normal(0, 1)
    p = gaussian.cdf(-eps)  # cdf(x) := P[X ≤ x], so this is P[x ≤ -εσ], where σ = 1

    if grid == "quantile":  # quantile(p) := {x | P[X ≤ x] = p}
        # Linear in cumulative probability (according to latent prior, which is subtly wrong; see Lin et al.)
        grid_x = gaussian.quantile(np.linspace(p, 1 - p, n)).numpy()
        grid_y = gaussian.quantile(np.linspace(p, 1 - p, n)).numpy()
    else:  # grid == "linear":
        # Linear in latent space coordinates
        grid_x = np.linspace(gaussian.quantile(p), gaussian.quantile(1 - p), n)
        grid_y = np.linspace(gaussian.quantile(p), gaussian.quantile(1 - p), n)

    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            # flip y so we can use origin="lower" for plotting the complete tiled image
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()[::-1, :]

    plt.figure(figno, figsize=(10, 10))
    plt.clf()
    plt.imshow(image, origin="lower", cmap="Greys_r")

    # Show latent space coordinates (center a tick on each row/column, labeled with the coordinate)
    startx = digit_size / 2
    endx = image_width - (digit_size / 2)
    tick_positions_x = np.array(startx + np.linspace(0, 1, len(grid_x)) * (endx - startx), dtype=int)
    tick_positions_y = tick_positions_x
    plt.xticks(tick_positions_x, [f"{x:0.3g}" for x in grid_x])
    plt.yticks(tick_positions_y, [f"{y:0.3g}" for y in grid_y])
    plt.xlabel(r"$z_{1}$")
    plt.ylabel(r"$z_{2}$")

    plt.title(f"Latent space ({grid} grid, up to ±{eps}σ)")
    plt.tight_layout()
    _create_directory(output_dir)
    plt.savefig(f"{output_dir}{latent_vis_basename}.{fig_format}")
    plt.draw()
    plotmagic.pause(0.1)  # force redraw

def main():
    # Make preparations

    _clear_and_create_directory(output_dir)

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_size = train_images.shape[0]  # 60k
    test_size = test_images.shape[0]  # 10k

    def preprocess_images(images, discrete=False):
        """Preprocess MNIST dataset."""
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.  # scale to [0, 1]
        if discrete:  # binarize to {0, 1}, to make compatible with discrete Bernoulli observation model
            images = np.where(images > .5, 1.0, 0.0)
        return images.astype("float32")

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

    # Train the model

    num_examples_to_generate = 16

    # Keeping the random vector constant for generation (prediction), it will be easier to see the improvement.
    # random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # Or... we can pick a sample of the test set for generating the corresponding reconstructed images.
    assert num_examples_to_generate <= batch_size
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    # Debug / info
    model.encoder.summary()
    model.decoder.summary()

    # Visualize the random initial state
    plot_and_save_epoch_image(model, 0, test_sample)
    plt.show()  # must call plt.show() once before pause works
    plotmagic.pause(0.001)  # force redraw

    # Train the model
    est = ETAEstimator(epochs, keep_last=10)
    with timer() as tim_total:
        for epoch in range(1, epochs + 1):
            # SGD using one pass through the training set (with the batches set up previously)
            with timer() as tim_train:
                for train_x in train_dataset:
                    train_step(model, train_x, optimizer)

            # For benchmarking: compute total ELBO on the test set
            with timer() as tim_test:
                running_mean = tf.keras.metrics.Mean()
                for test_x in test_dataset:
                    running_mean(compute_loss(model, test_x))
                elbo = -running_mean.result()

            plot_and_save_epoch_image(model, epoch, test_sample)
            est.tick()
            # dt_avg = sum(est.que) / len(est.que)
            print(f"Epoch: {epoch}, Test set ELBO: {elbo:0.6g}, epoch walltime training {tim_train.dt:0.3g}s, testing {tim_test.dt:0.3g}s; {est.formatted_eta}")
    print(f"Total training/testing wall time: {tim_total.dt:0.3g}s")

    # # TODO: Saving a CVAE instance using the official Keras serialization API doesn't work yet.
    # # Save the trained model.
    # # To reload the trained model in another session:
    # #   import keras
    # #   import main
    # #   main.model = keras.models.load_model("my_model")
    # # Now the model is loaded; you should be able to e.g.
    # #   main.plot_and_save_latent_image(20)
    # #
    # # force the model to build its graph to make it savable
    # dummy_data = tf.random.uniform((batch_size, 28, 28, 1))
    # _ = model(dummy_data)
    # model.save("my_model")
    model.my_save()  # this custom saving hack (saving the encoder/decoder separately) works

    # Make a gif animation of the training epochs
    with imageio.get_writer(f"{output_dir}{anim_filename}", mode="I") as writer:
        filenames = glob.glob(f"{output_dir}{fig_basename}*.{fig_format}")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)

    # Visualize latent representation
    plot_and_save_latent_image(20, figno=2)

if __name__ == "__main__":
    # To allow easy access to our global-scope variables in the live REPL session,
    # we make the main module (this module) available as `main` in the REPL scope.
    import sys
    repl_server.start(locals={"main": sys.modules["__main__"]})

    plt.ion()
    main()
    plt.ioff()
    plt.show()
