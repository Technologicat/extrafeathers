#!/usr/bin/python
#
# Convolutional variational autoencoder implemented in Keras and TensorFlow.
#
# Based on combining material from these two tutorials:
#   https://www.tensorflow.org/tutorials/generative/cvae
#   https://keras.io/examples/generative/vae/

# Start a REPL server (in main()) so we can inspect/save global-scope variables while the process is live.
# This is convenient if we have forgotten to include some save command in the script before starting it.
# To connect, `python -m unpythonic.net.client localhost`.
import sys
import unpythonic.net.server as repl_server

# TODO: configure paths to conform to other extrafeathers demos
# TODO: use plotmagic.pause (see euleriansolid)
# TODO: change the decoder model to a Gaussian (with learnable variance) as suggested in the paper by Lin et al.

import glob

from unpythonic import timer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

# import PIL
import imageio

# --------------------------------------------------------------------------------
# Config

# batch_size = 32  # for CPU
# batch_size = 128  # faster on GPU, still acceptable generalization on discrete Bernoulli
batch_size = 64  # acceptable generalization on continuous Bernoulli

latent_dim = 2  # use a 2-dimensional latent space so that we can easily visualize the results

epochs = 40

anim_filename = 'cvae.gif'
latent_image_filename = 'latent.png'

# For a discussion of NN optimization methods, see the Deep Learning book by Goodfellow et al.
optimizer = tf.keras.optimizers.Adam(1e-4)

# --------------------------------------------------------------------------------
# NN architecture

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Modified from https://keras.io/examples/generative/vae/
        # Encoder differences:
        #   - No z value in output, we use the reparameterize function instead.
        #     The custom Sampling layer in the original is a neat technique,
        #     but this is conceptually cleaner (encoder output is explicitly
        #     a distribution (represented as parameters), not a sample from it).
        # Decoder differences:
        #   - We handle the final (sigmoid) activation of the decoder manually later.
        #   - We have added an extra Dense layer of 16 units at the input side
        #     to more closely mirror the structure of the encoder.
        #   - The architecture is an exact mirror image of the encoder.
        encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                                   strides=2, padding="same")(encoder_inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                                   strides=2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        # https://linux-blog.anracom.com/2022/10/23/variational-autoencoder-with-tensorflow-2-8-xii-save-some-vram-by-an-extra-dense-layer-in-the-encoder/
        x = tf.keras.layers.Dense(units=16, activation="relu")(x)
        z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var")(x)
        encoder_outputs = [z_mean, z_log_var]
        self.encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
        self.encoder.summary()

        # decoder - exact mirror image of encoder (w.r.t. tensor sizes at each step)
        decoder_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(units=16, activation="relu")(decoder_inputs)
        x = tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu")(x)
        x = tf.keras.layers.Reshape(target_shape=(7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
                                            strides=2, padding="same")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same")(x)
        self.decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()

    @tf.function
    def sample(self, z=None):
        """Sample from the observation model pθ(x|z), returning the pixel-wise means.

        `z`: tf array of size `(batch_size, latent_dim)`.
             If not specified, a batch of 100 random samples is returned.
        """
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)

    def encode(self, x):
        """Encode a batch of input data. Return the posterior parameters `(μ, log σ)`."""
        # mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        # return mean, logvar
        return self.encoder(x)

    def reparameterize(self, mean, logvar):
        """Map  (μ, log σ) → z  stochastically.

        Return (ε, z) (the same ε sample is needed in the ELBO computation).
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps, eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """z → μ of p(x|z)"""
        # TODO: It is unnecessarily confusing to interpret the output as logits.
        # TODO: The encoder NN just produces real numbers, which we can interpret as we like.
        # TODO: How we use them determines what the optimization process drives them to become.
        logits = self.decoder(z)
        if apply_sigmoid:
            # For the discrete Bernoulli distribution, the mean is the same as the Bernoulli parameter,
            # which is the same as the probability of the output taking on the value 1 (instead of 0).
            probs = tf.sigmoid(logits)
            return probs
        return logits

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
# which matches Kingma and Welling (2019).
def log_normal_pdf(x, mean, logvar, raxis=1):
    log2pi = tf.math.log(2 * np.pi)
    return tf.reduce_sum(-0.5 * ((x - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
                         axis=raxis)

# https://github.com/cunningham-lab/cb_and_cc
def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    # computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    # returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
    # [l_lim, u_lim].
    # cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
    # both options, regardless of the value of the condition.
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
    # This implementation is unnecessarily confusing; logits and the binary cross-entropy are a sideshow,
    # specific to the Bernoulli observation model of the classic VAE, which doesn't even make sense for
    # continuous (grayscale) data (hence the unnecessary binarization of the input data, which hurts quality).
    #
    # How to compute pθ(x|z) in general, for any VAE: we just take the observation parameters P computed by
    # P = NN_dec(θ, z) at the sampled z (thus accounting for the dependence on z), and evaluate the known
    # function pθ(x|z) (parameterized by P and x) with those parameters P at the input x.
    #
    # Note that in a VAE, unlike a classical AE, the decoded output is not directly x-hat (i.e. an
    # approximation of the input data point), but parameters for a distribution that can be used to
    # compute x-hat.
    #
    # For more details, see the VAE tutorial by Kingma and Welling (2019, algorithm 2).
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
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=P, labels=x)
    # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # # discrete Bernoulli, explicitly
    # p = tf.sigmoid(P)  # interpret decoder output as logits; map into probabilities
    # p = tf.clip_by_value(p, 1e-4, 1 - 1e-4)  # avoid log(0)
    # logpx_z = tf.reduce_sum(x * tf.math.log(p) + (1 - x) * tf.math.log(1 - p),
    #                         axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # continuous Bernoulli - add a log-normalizing constant to make the probability distribution sum to 1
    # https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/cb_vae_mnist.ipynb
    p = tf.sigmoid(P)  # interpret decoder output as logits; map into probabilities
    p = tf.clip_by_value(p, 1e-4, 1 - 1e-4)  # avoid log(0)
    logpx_z = tf.reduce_sum(x * tf.math.log(p) + (1 - x) * tf.math.log(1 - p) + cont_bern_log_norm(p),
                            axis=[1, 2, 3])  # log pθ(x|z) (observation model)

    # We choose the latent prior pθ(z) to be a spherical unit Gaussian, N(0, 1).
    # Note the function `log_normal_pdf` takes `log σ`, not bare σ.
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

# TODO: currently only works for test sample size 16
def generate_and_save_epoch_image(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    ignored_eps, z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    plt.figure(1, figsize=(8, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 8, 1 + (2 * i))
        plt.imshow(test_sample[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.subplot(4, 8, 1 + (2 * i + 1))
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""

    gaussian = tfp.distributions.Normal(0, 1)
    grid_x = gaussian.quantile(np.linspace(0.001, 0.999, n))  # 3σ
    grid_y = gaussian.quantile(np.linspace(0.001, 0.999, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(2, figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(latent_image_filename)

def main():
    # Load the data
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_size = train_images.shape[0]  # 60k
    test_size = test_images.shape[0]  # 10k

    def preprocess_images(images, discrete=False):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.  # scale to [0, 1]
        if discrete:  # binarize to {0, 1}, to make compatible with discrete Bernoulli observation model
            images = np.where(images > .5, 1.0, 0.0)
        return images.astype('float32')

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

    # Train the model

    # Keeping the random vector constant for generation (prediction),
    # it will be easier to see the improvement.
    num_examples_to_generate = 16
    # random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # Or... we can pick a sample of the test set for generating the corresponding output images
    assert num_examples_to_generate <= batch_size
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    # Visualize the random initial state
    generate_and_save_epoch_image(model, 0, test_sample)

    # Train the model
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

            # display.clear_output(wait=False)  # Jupyter?
            print('Epoch: {}, Test set ELBO: {}, time elapsed for current epoch: training {}, testing {}'
                  .format(epoch, elbo, tim_train.dt, tim_test.dt))
            generate_and_save_epoch_image(model, epoch, test_sample)
    print(f'Total time elapsed for training: {tim_total.dt} seconds')
    model.save_weights("weights.h5")

    # Make a gif animation of the training epochs
    with imageio.get_writer(anim_filename, mode='I') as writer:
        filenames = glob.glob('image_at_epoch*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)

    # Visualize latent representation
    plot_latent_images(model, 20)

if __name__ == '__main__':
    # To allow easy access to our global-scope variables in the live REPL session,
    # we make the main module (this module) available as `main` in the REPL scope.
    repl_server.start(locals={"main": sys.modules["__main__"]})
    plt.ion()
    main()
    plt.ioff()
    plt.show()
