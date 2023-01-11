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
  main.plot_latent_image(10)
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

# TODO: use an early-stopping criterion to avoid overfitting the training set?
# TODO: also save model snapshots so we can stop manually

# TODO: conform better to the Keras OOP API (right now we have a Model that doesn't behave as the is-a implies)
#  - for a custom Keras object, `train_step` should be a method, not a separate function
#  - `call` (although somewhat useless for an autoencoder) should be implemented
#  - how to support `fit` and `predict`?
#  - saving/serialization with standard API
# TODO: register some monitored metrics? This would allow using the `EarlyStopping` class from the Keras API.
# TODO: implement metrics from "Early stopping but when?" paper

# TODO: For use with PDE solution fields:
#   - Project the PDE solution to a uniform grid (uniform, since we use a convolutional NN)
#   - Change the decoder model to a Gaussian (with learnable variance), as suggested in the paper by Lin et al.
#   - Or better yet, first check the distribution of the actual fields (observed data!)

import glob
import os
import pathlib
import typing

from unpythonic import ETAEstimator, timer
from unpythonic.env import env

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

epochs = 500

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
        P = self.decode(z)
        p = tf.sigmoid(P)  # probability for discrete Bernoulli; λ parameter of continuous Bernoulli
        # For the discrete Bernoulli distribution, the mean is the same as the Bernoulli parameter,
        # which is the same as the probability of the output taking on the value 1 (instead of 0).
        # For the continuous Bernoulli distribution, we just return λ as-is (works fine as output in practice).
        return p

    def encode(self, x):
        """x → parameters of variational posterior qϕ(z|x), namely `(μ, log σ)`."""
        # # If we had a single output layer of double the size, we could do it like this:
        # mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        # return mean, logvar
        # But having two outputs explicitly, we can just:
        return self.encoder(x)

    def reparameterize(self, mean, logvar):
        """Map `(μ, log σ) → z` stochastically.

        Draw a single noise sample `ε`; return `(ε, z)`.

        This is the reparameterization trick, isolating the stochasticity of `z`
        into a non-parametric noise variable `ε`. Writing `z = g(μ(ϕ, x), [log σ](ϕ, x), ε)`
        allows keeping the transformation `g` deterministic, thus allowing backpropagation
        through graph nodes involving `z`.
        """
        # https://datascience.stackexchange.com/questions/51086/valueerror-cannot-convert-a-partially-known-tensorshape-to-a-tensor-256
        # `tf.shape` is dynamic; `mean.shape` is static (see docstring of `tf.shape`) and
        # does not work if one of the dimensions is not yet known (e.g. the batch size).
        # This is important to make the model saveable, because `Model.save` calls the
        # `call` method with an unspecified batch size (to build the computational graph).
        #
        # The noise sample is drawn from a unit spherical Gaussian, distinct from the latent prior.
        #
        # eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        return eps, z

    def decode(self, z):
        """z → parameters of observation model pθ(x|z)"""
        return self.decoder(z)

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
    #     P = self.decode(z)
    #     xhat = tf.sigmoid(P)
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
    """VAE loss function: negative of the ELBO, for a data batch `x`.

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
    # It is correct that even in the general case, we do want to minimize cross-entropy, but the ELBO is
    # easier to understand without introducing this extra concept.
    #
    # See e.g. Wikipedia on cross-entropy:
    #    https://en.wikipedia.org/wiki/Cross_entropy
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

    # Continuous Bernoulli - add a log-normalizing constant to make the probability distribution sum to 1,
    # when x is continuous in the interval [0, 1] (instead of taking on just one of the values {0, 1}).
    #
    # Note the output is now the parameter λ of the continuous Bernoulli distribution, not directly a
    # probability; and the mean of the continuous Bernoulli distribution is also different from λ.
    # Still, in practice λ works well as-is as a deterministic output value for the decoder.
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
    return loss

# --------------------------------------------------------------------------------
# Plotting helpers

def plot_test_sample_image(model: CVAE, epoch: int, test_sample: tf.Tensor, figno: int = 1) -> None:
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

    plt.figure(figno)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # prevent axes crawling
    ax.imshow(image, cmap="Greys_r")
    ax.axis("off")

    ax.set_title(f"Test sample, epoch {epoch} (left: input $\\mathbf{{x}}$, right: prediction $\\hat{{\\mathbf{{x}}}}$)")
    fig.tight_layout()

    plt.draw()
    plotmagic.pause(0.1)  # force redraw


def normal_grid(n: int = 20, kind: str = "quantile", eps: float = 3):
    """Make a grid on `[-εσ, +εσ]` for evaluating normally distributed quantities.

    μ = 0, σ = 1; shift and scale the result manually if necessary.

    `n`: number of points

    `grid`: grid spacing type; one of "linear" or "quantile" (default)

            "quantile" has normally distributed density, placing more emphasis
            on the region near the origin, where most of the gaussian probability
            mass is concentrated. This grid is linear in cumulative probability.

            "linear" is just a linear spacing. It effectively emphasizes the faraway
            regions, since the gaussian does not have much probability mass there.

    `eps`:  ε for lower/upper limit ±εσ. E.g. the default 3 means ±3σ.
    """
    assert kind in ("linear", "quantile")

    gaussian = tfp.distributions.Normal(0, 1)
    pmin = gaussian.cdf(-eps)  # cdf(x) := P[X ≤ x], so this is P[x ≤ -εσ], where σ = 1

    if kind == "quantile":  # quantile(p) := {x | P[X ≤ x] = p}
        # xx = gaussian.quantile(np.linspace(p, 1 - p, n)).numpy()  # yields +inf at ≥ +6σ
        xx = gaussian.quantile(np.linspace(pmin, 0.5, n // 2 + 1)).numpy()
        xx_left = xx[:-1]
        xx_right = -xx_left[::-1]
        if n % 2 == 0:
            xx = np.concatenate((xx_left, xx_right))
        else:
            xx = np.concatenate((xx_left, [0.0], xx_right))
    else:  # kind == "linear":
        xmin = gaussian.quantile(pmin)
        xmax = -xmin
        xx = np.linspace(xmin, xmax, n)

    assert np.shape(xx)[0] == n
    return xx


def plot_latent_image(n: int = 20, model: typing.Optional[CVAE] = None, digit_size: int = 28,
                      grid: str = "quantile", eps: float = 3, figno: int = 1,
                      epoch: typing.Optional[int] = None) -> env:
    """Plot n × n digit images decoded from the latent space.

    `n`, `grid`, `eps`: passed to `normal_grid` (`grid` is the `kind`)

                        A quantile grid is linear in cumulative probability according to the
                        latent prior. However, using the prior is subtly wrong, and the marginal
                        posterior of z should be used instead; see Lin et al.

    `digit_size`: width/height of each digit image (square-shaped), in pixels.
                  Must match what the model was trained for.
    `model`: `CVAE` instance, or `None` to use the default instance.
    `figno`: matplotlib figure number.
    `epoch`: if specified, included in the figure title.
    """
    if model is None:
        def get_default_model():
            global model
            return model
        model = get_default_model()
    assert isinstance(model, CVAE)

    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    zz = normal_grid(n, grid, eps)
    grid_x = zz
    grid_y = zz

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            # flip y so we can use origin="lower" for plotting the complete tiled image
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()[::-1, :]

    plt.figure(figno)
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.cla()
    plt.sca(ax)
    fig.tight_layout()  # <-- important to do this also here to prevent axes crawling
    ax.imshow(image, origin="lower", cmap="Greys_r")
    # print(ax._position.bounds)  # DEBUG

    # Show latent space coordinates (center a tick on each row/column, labeled with the coordinate)
    startx = digit_size / 2
    endx = image_width - (digit_size / 2)
    tick_positions_x = np.array(startx + np.linspace(0, 1, len(grid_x)) * (endx - startx), dtype=int)
    tick_positions_y = tick_positions_x
    ax.set_xticks(tick_positions_x, [f"{x:0.3g}" for x in grid_x], rotation="vertical")
    ax.set_yticks(tick_positions_y, [f"{y:0.3g}" for y in grid_y])

    ax.set_xlabel(r"$z_{1}$")
    ax.set_ylabel(r"$z_{2}$")

    epoch_str = f"; epoch {epoch}" if epoch is not None else ""
    ax.set_title(f"Latent space ({grid} grid, up to ±{eps}σ){epoch_str}")

    fig.tight_layout()

    plt.draw()
    plotmagic.pause(0.1)  # force redraw

    return env(n=n, model=model, digit_size=digit_size, grid=grid, eps=eps, figno=figno)


def overlay_datapoints(x: tf.Tensor, labels: tf.Tensor, figdata: env, alpha: float = 0.1) -> None:
    """Overlay the codepoints corresponding to a dataset `x` and `labels` onto the latent space plot.

    `figdata`: metadata describing the figure on which to overlay the plot.
               This is the return value of `plot_latent_image`.
    `alpha`: opacity of the scatterplot points.
    """
    n = figdata.n
    model = figdata.model
    digit_size = figdata.digit_size
    grid = figdata.grid
    eps = figdata.eps
    figno = figdata.figno

    assert isinstance(model, CVAE)

    # Find latent distribution parameters for the given data.
    # We'll plot the means.
    mean, logvar = model.encode(x)

    # We need some gymnastics to plot on top of an imshow image; it's easiest to
    # overlay a new Axes with a transparent background.
    # https://stackoverflow.com/questions/16829436/overlay-matplotlib-imshow-with-line-plots-that-are-arranged-in-a-grid
    plt.figure(figno)
    fig = plt.gcf()
    # axs = fig.axes  # list of all Axes objects in this Figure
    ax = plt.gca()
    plt.draw()  # force update of extents
    # box = ax._position.bounds  # whole of the Axes `ax`, in figure coordinates

    # print([int(x) for x in fig.axes[0].get_xlim()])

    # Compute position for overlay:
    #
    # Determine the centers of images at two opposite corners of the sheet,
    # in data coordinates of the imshow plot.
    image_width = digit_size * n
    xmin = digit_size / 2
    xmax = image_width - (digit_size / 2)
    ymin = xmin
    ymax = xmax
    xy0 = [xmin, ymin]
    xy1 = [xmax, ymax]

    # Convert to figure coordinates.
    def data_to_fig(xy):
        """Convert Matplotlib data coordinates (of current axis) to figure coordinates."""
        # https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
        xy_ax = ax.transLimits.transform(xy)  # data coordinates -> axes coordinates
        xy_disp = ax.transAxes.transform(xy_ax)  # axes -> display
        xy_fig = fig.transFigure.inverted().transform(xy_disp)  # display -> figure
        # print(f"data: {xy}")
        # print(f"ax:   {xy_ax}")
        # print(f"disp: {xy_disp}")
        # print(f"fig:  {xy_fig}")
        return xy_fig
    x0, y0 = data_to_fig(xy0)
    x1, y1 = data_to_fig(xy1)

    # Set up the new Axes, no background (`set_axis_off`), and plot the overlay.
    box = [x0, y0, (x1 - x0), (y1 - y0)]
    newax = fig.add_axes(box)
    newax.set_axis_off()

    # # Instead of using a global alpha, we could also customize a colormap like this
    # # (to make alpha vary as a function of the data value):
    # import matplotlib as mpl
    # rgb_colors = mpl.colormaps.get("viridis").colors  # or some other base colormap; or make a custom one
    # rgba_colors = [[r, g, b, alpha] for r, g, b in rgb_colors]
    # my_cmap = mpl.colors.ListedColormap(rgba_colors, name="viridis_translucent")
    # # mpl.colormaps.register(my_cmap, force=True)  # no need to register it as we can pass it directly.

    if grid == "quantile":
        # Invert the quantile spacing numerically, to make the positioning match the example images.
        # TODO: implement a custom ScaleTransform for data-interpolated axes? Useful both here and in `hdrplot`.
        n_interp = 10001
        raw_zi = normal_grid(n_interp, grid, eps)  # data value
        linear_zi = np.linspace(-eps, eps, n_interp)  # where that value is on a display with linear coordinates
        def to_linear_display_coordinate(zi):
            """raw value of z_i -> display position on a linear axis with interval [-eps, eps]"""
            return np.interp(zi, xp=raw_zi, fp=linear_zi, left=np.nan, right=np.nan)  # nan = don't plot
        linear_z1 = to_linear_display_coordinate(mean[:, 0])
        linear_z2 = to_linear_display_coordinate(mean[:, 1])
    else:  # grid == "linear":
        linear_z1 = mean[:, 0]
        linear_z2 = mean[:, 1]

    newax.scatter(linear_z1, linear_z2, c=labels, alpha=alpha)
    # newax.scatter(linear_z1, linear_z2, c=labels, cmap=my_cmap)
    # newax.patch.set_alpha(0.25)  # patch = Axes background
    newax.set_xlim(-eps, eps)
    newax.set_ylim(-eps, eps)
    plt.draw()
    plotmagic.pause(0.1)  # force redraw


# --------------------------------------------------------------------------------
# Main program - model training

def preprocess_images(images, discrete=False):
    """Preprocess MNIST dataset."""
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.  # scale to [0, 1]
    if discrete:  # binarize to {0, 1}, to make compatible with discrete Bernoulli observation model
        images = np.where(images > .5, 1.0, 0.0)
    # else continuous grayscale data
    return images.astype("float32")

# For overlay plotting, it's convenient to have these here.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

def main():
    # Make preparations

    _clear_and_create_directory(output_dir)

    # Set up figures
    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))  # test example
    fig1.tight_layout()
    fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))  # ELBO history
    fig2.tight_layout()
    fig3, axs3 = plt.subplots(1, 1, figsize=(10, 10))  # latent space
    fig3.tight_layout()

    # must call `plt.show` once before `plotmagic.pause` works
    plt.show()
    plt.draw()
    plotmagic.pause(0.001)

    train_size = train_images.shape[0]  # 60k
    test_size = test_images.shape[0]  # 10k

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

    # Plot the random initial state
    plot_test_sample_image(model, 0, test_sample)
    plot_test_sample_image(model, 0, test_sample)  # and again to prevent axes crawling
    plt.savefig(f"{output_dir}{fig_basename}_0000.{fig_format}")

    e = plot_latent_image(21, figno=3, epoch=0)
    e = plot_latent_image(21, figno=3, epoch=0)  # and again to prevent axes crawling
    plt.savefig(f"{output_dir}{latent_vis_basename}_0000.{fig_format}")

    # Train the model
    est = ETAEstimator(epochs, keep_last=10)
    train_elbos = []
    test_elbos = []
    with timer() as tim_total:
        for epoch in range(1, epochs + 1):
            # SGD using one pass through the training set (with the batches set up previously)
            with timer() as tim_train:
                running_mean = tf.keras.metrics.Mean()
                for train_x in train_dataset:
                    running_mean(train_step(model, train_x, optimizer))
                train_elbo = -running_mean.result()
                train_elbos.append(train_elbo)

            # For benchmarking: compute total ELBO on the test set
            with timer() as tim_test:
                running_mean = tf.keras.metrics.Mean()
                for test_x in test_dataset:
                    running_mean(compute_loss(model, test_x))
                test_elbo = -running_mean.result()
                test_elbos.append(test_elbo)

            plot_test_sample_image(model, epoch, test_sample)
            plt.savefig(f"{output_dir}{fig_basename}_{epoch:04d}.{fig_format}")

            # plot ELBO and save the ELBO history (for visual tracking of training)
            plt.figure(2)
            fig = plt.gcf()
            ax = fig.axes[0]
            ax.cla()
            plt.sca(ax)
            xx = np.arange(1, epoch + 1)
            ax.plot(xx, train_elbos, label="train")
            ax.plot(xx, test_elbos, label="test")
            ax.xaxis.grid(visible=True, which="both")
            ax.yaxis.grid(visible=True, which="both")
            ax.set_xlabel("epoch")
            ax.set_ylabel("ELBO")
            ax.legend(loc="best")
            fig.tight_layout()
            plt.draw()
            plotmagic.pause(0.1)
            plt.savefig(f"{output_dir}elbo.{fig_format}")
            np.savez(f"{output_dir}elbo.npz", epochs=xx, train_elbos=train_elbos, test_elbos=test_elbos)

            # Plot and save latent representation
            e = plot_latent_image(21, figno=3, epoch=epoch)
            plt.savefig(f"{output_dir}{latent_vis_basename}_{epoch:04d}.{fig_format}")
            # overlay_datapoints(train_images, train_labels, e)  # out of memory on GPU

            # snapshot model every now and then
            if epoch % 5 == 1:
                model.my_save()

            est.tick()
            # dt_avg = sum(est.que) / len(est.que)
            print(f"Epoch: {epoch}, training set ELBO {train_elbo:0.6g}: test set ELBO {test_elbo:0.6g}, epoch walltime training {tim_train.dt:0.3g}s, testing {tim_test.dt:0.3g}s; {est.formatted_eta}")
    print(f"Total training/testing wall time: {tim_total.dt:0.3g}s")

    # # TODO: Saving a CVAE instance using the official Keras serialization API doesn't work yet.
    # # Save the trained model.
    # # To reload the trained model in another session:
    # #   import keras
    # #   import main
    # #   main.model = keras.models.load_model("my_model")
    # # Now the model is loaded; you should be able to e.g.
    # #   main.plot_latent_image(20)
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
    e = plot_latent_image(21, figno=3)
    plt.savefig(f"{output_dir}{latent_vis_basename}_final.{fig_format}")

    # ...and once again with a training dataset overlay
    overlay_datapoints(train_images, train_labels, e)
    plt.savefig(f"{output_dir}{latent_vis_basename}_annotated.{fig_format}")

if __name__ == "__main__":
    # To allow easy access to our global-scope variables in the live REPL session,
    # we make the main module (this module) available as `main` in the REPL scope.
    import sys
    repl_server.start(locals={"main": sys.modules["__main__"]})

    plt.ion()
    main()
    plt.ioff()
    plt.show()
