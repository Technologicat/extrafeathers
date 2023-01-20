"""The CVAE model."""

__all__ = ["CVAE",
           "compute_loss",
           "train_step"]

import pathlib
import numpy as np
import tensorflow as tf

from .config import latent_dim
from .resnet import (IdentityBlock2D, IdentityBlockTranspose2D,
                     ConvolutionBlock2D, ConvolutionBlockTranspose2D)
from .util import clear_and_create_directory

# --------------------------------------------------------------------------------
# NN architecture

extra_layer_size = 16

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

    # # classical VAE
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
    #                            strides=2, padding="same")(encoder_inputs)     # 28×28×1 → 14×14×32
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
    #                            strides=2, padding="same")(x)                  # 14×14×32 → 7×7×64

    # # ResNet attempt 1 (performs about as well as attempt 2)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
    #                            strides=2, padding="same")(encoder_inputs)     # 28×28×1 → 14×14×32
    # x = IdentityBlock2D(filters=32, kernel_size=3, bottleneck_factor=1)(x)    # 14×14×32→ 14×14×32
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
    #                            strides=2, padding="same")(x)                  # 14×14×32 → 7×7×64
    # x = IdentityBlock2D(filters=64, kernel_size=3, bottleneck_factor=1)(x)    # 7×7×64 → 7×7×64

    # # # ResNet attempt 2 - large shallow model, good results
    # x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu",
    #                        bottleneck_factor=1)(encoder_inputs)  # 28×28×1 → 14×14×32
    # x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
    #                     bottleneck_factor=1)(x)                  # 14×14×32→ 14×14×32
    # x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu",
    #                        bottleneck_factor=1)(x)               # 14×14×32 → 7×7×64
    # x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
    #                     bottleneck_factor=1)(x)                  # 7×7×64 → 7×7×64

    # # ResNet attempt 3 - default bottleneck factor of 4, smaller model, but more blurred output
    # x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu")(encoder_inputs)  # 28×28×1 → 14×14×32
    # x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
    # x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu")(x)               # 14×14×32 → 7×7×64
    # x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu")(x)                  # 7×7×64 → 7×7×64

    # # ResNet attempt 4
    # x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu")(encoder_inputs)  # 28×28×1 → 14×14×32
    # x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
    # x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
    # x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu")(x)               # 14×14×32 → 7×7×64
    # x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu")(x)                  # 7×7×64 → 7×7×64
    # x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu")(x)                  # 7×7×64 → 7×7×64

    # ResNet attempt 5
    x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu",
                           bottleneck_factor=2)(encoder_inputs)  # 28×28×1 → 14×14×32
    x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                        bottleneck_factor=2)(x)                  # 14×14×32→ 14×14×32
    x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                        bottleneck_factor=2)(x)                  # 14×14×32→ 14×14×32
    x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu",
                           bottleneck_factor=2)(x)               # 14×14×32 → 7×7×64
    x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                        bottleneck_factor=2)(x)                  # 7×7×64 → 7×7×64
    x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                        bottleneck_factor=2)(x)                  # 7×7×64 → 7×7×64

    x = tf.keras.layers.Flatten()(x)
    # VRAM saving trick from the Keras example: the encoder has *two* outputs: mean and logvar. Hence,
    # if we add a small dense layer after the convolutions, and connect that to the outputs, we will have
    # much fewer trainable parameters (in total) than if we connect the last convolution layer to the outputs
    # directly. As pointed out by:
    # https://linux-blog.anracom.com/2022/10/23/variational-autoencoder-with-tensorflow-2-8-xii-save-some-vram-by-an-extra-dense-layer-in-the-encoder/
    #
    # Note that this trick only helps if `2 * latent_dim > extra_layer_size`; otherwise the architecture
    # with the extra layer uses *more* VRAM. However, in that scenario it increases the model capacity slightly
    # (but in practice we get better results if we add more layers in the convolution part instead of making
    # this one larger).
    x = tf.keras.layers.Dense(units=extra_layer_size, activation="relu")(x)
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
    x = tf.keras.layers.Dense(units=extra_layer_size, activation="relu")(decoder_inputs)
    x = tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu")(x)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)

    # Now we are at the point of the architecture where we have the Conv2D output,
    # so let's mirror the encoder architecture to return to the input space.
    #
    # Note no activation function in the output layer - we want arbitrary real numbers as output.
    # The output will be interpreted as parameters `P` for the observation model pθ(x|z).
    # Here we want just something convenient that we can remap as necessary.

    # # Classical VAE
    # x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
    #                                     strides=2, padding="same")(x)     # 7×7×64 → 14×14×32
    # decoder_outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2,
    #                                                   padding="same")(x)  # 14×14×32 → 28×28×1

    # # ResNet attempt 1 (performs about as well as attempt 2)
    # x = IdentityBlockTranspose2D(filters=64, kernel_size=3, bottleneck_factor=1)(x)    # 7×7×64 → 7×7×64
    # x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
    #                                     strides=2, padding="same")(x)                  # 7×7×64 → 14×14×32
    # x = IdentityBlockTranspose2D(filters=32, kernel_size=3, bottleneck_factor=1)(x)    # 14×14×32 → 14×14×32
    # decoder_outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3,
    #                                                   strides=2, padding="same")(x)    # 14×14×32 → 28×28×1

    # # ResNet attempt 2 - large shallow model, good results
    # x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu",
    #                              bottleneck_factor=1)(x)                   # 7×7×64 → 7×7×64
    # x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
    #                                 bottleneck_factor=1)(x)                # 7×7×64 → 14×14×32
    # x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
    #                              bottleneck_factor=1)(x)                   # 14×14×32 → 14×14×32
    # decoder_outputs = ConvolutionBlockTranspose2D(filters=1, kernel_size=3,
    #                                               bottleneck_factor=1)(x)  # 14×14×32 → 28×28×1

    # # ResNet attempt 3 - default bottleneck factor of 4, smaller model, but more blurred output
    # x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
    # x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)  # 7×7×64 → 14×14×32
    # x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
    # decoder_outputs = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)        # 14×14×32 → 28×28×1

    # # ResNet attempt 4
    # x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
    # x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
    # x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)  # 7×7×64 → 14×14×32
    # x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
    # x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
    # decoder_outputs = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)        # 14×14×32 → 28×28×1

    # ResNet attempt 5
    x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu",
                                 bottleneck_factor=2)(x)     # 7×7×64 → 7×7×64
    x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu",
                                 bottleneck_factor=2)(x)     # 7×7×64 → 7×7×64
    x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                    bottleneck_factor=2)(x)  # 7×7×64 → 14×14×32
    x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                 bottleneck_factor=2)(x)     # 14×14×32 → 14×14×32
    x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                 bottleneck_factor=2)(x)     # 14×14×32 → 14×14×32
    decoder_outputs = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)        # 14×14×32 → 28×28×1

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
    def my_save(self, path: str):
        clear_and_create_directory(path)
        p = pathlib.Path(path).expanduser().resolve()
        self.encoder.save(str(p / "encoder"))
        self.decoder.save(str(p / "decoder"))
    def my_load(self, path: str):
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

@tf.function
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

    # Optional multiple-sample Monte Carlo just for the lulz.
    elbo = 0.0
    n_mc_samples = 1
    for _ in range(n_mc_samples):
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

        # Finally, evaluate the ELBO. We have
        #
        #   ELBO[θ,ϕ](x) := E[qϕ(z|x)] [log pθ(x,z) - log qϕ(z|x)]
        #                 = E[qϕ(z|x)] [log (pθ(x|z) pθ(z)) - log qϕ(z|x)]       (rewrite joint probability)
        #                 = E[qϕ(z|x)] [log pθ(x|z) + log pθ(z) - log qϕ(z|x)]   (log arithmetic)
        #                 = E[p(ε)] [log pθ(x|z) + log pθ(z) - log qϕ(z|x)]      (reparameterization trick)
        #                 ≃ log pθ(x|z) + log pθ(z) - log qϕ(z|x)
        #
        # where E[var][expr] is the expectation, and the simeq symbol ≃ denotes that one side
        # (here the right-hand side) is an unbiased estimator of the other side.
        #
        # For understanding how the ELBO works, consider
        #
        #   ELBO[θ,ϕ](x) ≡ E[qϕ(z|x)] [log pθ(x,z) - log qϕ(z|x)]
        #                = E[qϕ(z|x)] [log (pθ(x|z) pθ(z)) - log qϕ(z|x)]
        #                = E[qϕ(z|x)] [log pθ(x|z) - [log qϕ(z|x) - log pθ(z)]]
        #                ≡ E[qϕ(z|x)] [log pθ(x|z)] - DKL(qϕ(z|x) ‖ pθ(z))
        #
        # where DKL is the Kullback--Leibler divergence. So the above expression is the sum of
        # a reconstruction quality term and a regularization term. On the other hand, by rewriting
        # the joint probability the other way,
        #
        #   ELBO[θ,ϕ](x) ≡ E[qϕ(z|x)] [log pθ(x,z) - log qϕ(z|x)]
        #                = E[qϕ(z|x)] [log (pθ(x) pθ(z|x)) - log qϕ(z|x)]
        #                = E[qϕ(z|x)] [log pθ(x) - [log qϕ(z|x) - log pθ(z|x)]]
        #                = E[qϕ(z|x)] [log pθ(x)] - E[qϕ(z|x)] [log qϕ(z|x) - log pθ(z|x)]]
        #                ≡ E[qϕ(z|x)] [log pθ(x)] - DKL(qϕ(z|x) ‖ pθ(z|x))
        #                ≡ log pθ(x) - DKL(qϕ(z|x) ‖ pθ(z|x))
        #
        # which, since  DKL ≥ 0,  highlights two important facts. First,
        #
        #   ELBO[θ,ϕ](x) ≤ log p(x)
        #
        # for all θ,ϕ; and secondly, the tightness of this inequality depends on how close the
        # approximate variational posterior qϕ(z|x) is to the unknown true posterior pθ(z|x).
        # So maximizing the ELBO should allow us to find the optimal qϕ(z|x), which approximates
        # pθ(z|x) best, within our chosen class of distributions qϕ(z|x).
        #
        # Keep in mind that for a continuous distribution p(x), the ELBO may actually take on positive
        # values, because p(x) is then a probability *density*, which may in general exceed 1.
        elbo += tf.reduce_mean(logpx_z + logpz - logqz_x)
    if n_mc_samples > 1:  # take the mean
        elbo /= n_mc_samples

    return -elbo  # with sign flipped → ELBO loss

@tf.function
def train_step(model, x, optimizer):
    """Execute one training step, computing and applying gradients via backpropagation."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
