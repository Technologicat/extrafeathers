"""The CVAE model (convolutional variational autoencoder). To compress is to understand."""

__all__ = ["CVAE",
           "elbo_loss",
           "active_units",
           "negative_log_likelihood"]

import pathlib

import numpy as np

import tensorflow as tf

from unpythonic import prod

from .config import latent_dim
from .resnet import (IdentityBlock2D, IdentityBlockTranspose2D,
                     ProjectionBlock2D, ProjectionBlockTranspose2D,
                     ConvolutionBlock2D, ConvolutionBlockTranspose2D,
                     GNDropoutRegularization)
from .util import clear_and_create_directory, batched

# --------------------------------------------------------------------------------
# NN architecture

# extra_layer_size = 16  # The extra layer is omitted in recent revisions (better performance, i.e. higher ELBO)
dropout_fraction = 0.1

# Encoder/decoder architecture modified from https://keras.io/examples/generative/vae/
#
# Encoder differences:
#   - No `z` value in output. We instead output `(μ, log σ)`, and introduce a separate
#     `reparameterize` function, which maps `(μ, log σ) → z` (by sampling). The custom
#     `Sampling` layer in the original example is a neat technique, but this is conceptually
#     cleaner, because now the encoder output is explicitly a distribution (represented as
#     parameters), not a sample from it.
#
# Decoder differences:
#   - We handle the final (sigmoid) activation of the decoder manually later.
#     Similarly, this is conceptually cleaner; the decoder output is now parameters
#     for a distribution, not a sample from it. The decoder output is just a tensor
#     of pixel-wise real numbers `P`. Choosing to use that data as `λ = sigmoid(P)`,
#     in a continuous Bernoulli distribution, and also choosing to output the `λ`
#     tensor directly as the output picture, are separate design decisions.
#   - We have added an extra Dense layer of `extra_layer_size` units at the input
#     side of the decoder to more closely mirror the structure of the output side
#     of the encoder.
#   - The architecture is now an exact mirror image of the encoder, with one exception:
#     the input to the decoder is a code point `z`, not the distribution parameters
#     `(μ, log σ)`. (We would need to reduce it into a sample anyway when computing
#     the Monte Carlo approximation to the ELBO loss.)
#
# Note the encoder/decoder don't need to be mirror images. Actually, in a VAE they are
# completely separate entities, and can be *anything* that is applicable to the desired job.
# The only point of connection between them is the latent representation. All we need are
# two highly nonlinear functions:
#   - Encoder: takes an input picture and extracts its features into a code.
#   - Decoder: takes such a code, and draws the corresponding picture.
# The reason for packaging these functions into a VAE for joint training is that this
# forces the learned code to become Lipschitz-continuous (with a reasonably small Lipschitz parameter),
# i.e. makes it so that *any* small change in the input picture maps to small change in the code point.
#
# The decoder can be thought of as a conditional generative model, conditioned on the code point `z`.
# This is somewhat like Stable Diffusion, but much smaller and more limited, the code point playing
# the role of the prompt. Also the encoder can be thought of as a relative of img2img, to get the latent
# representation for a given image. (Though since we don't add noise during inference, the learned mapping
# becomes deterministic - which is what makes this useful as a computational accelerator.)
#
# The decoder can be *anything*. So, how to compute `log pθ(x|z)` given just a black-box output? Fortunately,
# we can pretend - just like for the original "mirror-image" NN decoder, after its sigmoid activation - that
# the decoder output is not actually a picture, but the pixelwise λ parameter for the continuous Bernoulli distribution,
# and estimate `log pθ(x|z)` accordingly. This is no different from how the original version works; and it works
# as long as each pixel in the input data is approximately continuous-Bernoulli distributed; which should be true
# for any picture (at least in the classical low dynamic range regime). Non-pictorial data (such as PDE solution fields)
# may require tweaking the observation model.

# See e.g. the study by Dieng et al. (2019), which uses a ResNet as the encoder, and a Gated PixelCNN
# as the decoder:
#   https://arxiv.org/pdf/1807.04863.pdf
#
# And, why do we need a CVAE at all - why not directly use t-SNE, or some other classical dimension reduction method,
# to produce the latent? This is because the ranges of applicability are different. Methods such as t-SNE take a
# couple dozen features, and map them down to just a few. A CVAE takes *whole pictures*, which in real applications
# can be up to megapixels in size, and describes them in those couple dozen features.

# TODO: Parameterize the input data shape. The encoder is currently hardcoded for 28×28×1, a measly 768 pixels,
# TODO: and it spatially downsamples twice. Note also the decoder architecture, which upsamples twice.
#
# TODO: Improve the NN architecture?
#
# Implement a PINN (physically informed neural network) mechanism for penalizing the deviation of a decoded image
# from a valid numerical solution of a given PDE.
#
# We may need finite differences (to compute on GPU), or perhaps the Tikhonov-regularized derivative, if data is noisy; see e.g.:
#   https://ejde.math.txstate.edu/conf-proc/21/k3/knowles.pdf
#   https://www.researchgate.net/publication/255907171_Numerical_Differentiation_of_Noisy_Nonsmooth_Data
#   https://www.researchgate.net/profile/Rick-Chartrand/publication/321682456_Numerical_differentiation_of_noisy_nonsmooth_multidimensional_data
#
# Try various different kernel sizes?
#
# A spatial pyramid pooling (SPP) layer before the final fully connected layer(s) is also an
# option. This is especially useful for producing a fixed-length representation for varying input
# image sizes. But is there an inverse of SPP, for the decoder? (No, but strictly we don't need one;
# see above.)
#   https://arxiv.org/abs/1406.4729
#   https://github.com/yhenon/keras-spp
#
# Explore skip-connections more fully.
#   - At least two styles exist for re-using inputs of earlier layers: ResNet, and concatenation.
#   - Skip-connections from an encoder layer to the decoder layer of the same size?
#     - This kind of architecture is commonly seen in autoencoder designs used for approximating PDE solutions
#       (PINNs, physically informed neural networks); it should speed up training of the encoder/decoder combination
#       by allowing the decoded output to directly influence the weights on the encoder side.
#         - Using the functional API, we should be able to set up three `Model`s that share layer instances:
#           the encoder, the decoder, and (for training) the full autoencoder that combines both and adds the
#           skip-connections from the encoder side to the decoder side.
#         - But how to run such a network in decoder-only mode (i.e. generative mode), where the encoder layers are not available?
#           For real data, we could save the encoder layer states as part of the coded representation. But for generating new data, that's a no-go.
#     - OTOH, maybe we shouldn't do that; in unsupervised learning (such as in a VAE), where the goal is to explain and reconstruct the input
#       by applying a bottleneck simplification, the encoder and decoder should be thought of as separate entities. The goal of a VAE is representation
#       learning, not perfect reconstruction (which would be trivial if the skip-connections did all the work; this would make the learned representation
#       completely useless). This is different from supervised learning, such as in a U-Net, where the output is a segmentation map; crucially, in that case
#       the goal is NOT a reconstruction of the input! In such cases, connecting the similarly sized layers between the halves of the "U" does indeed help.
#           https://www.reddit.com/r/deeplearning/comments/ds1245/comment/f6mqy4q/
#       See the original U-Net paper by Ronneberger et al. (2015). The authors explain in the abstract that the "contracting path [...]
#       captures context", while the "symmetric expanding path" (connected with such cross-skip-connections) "enables precise localization"
#       of that context in the network output (i.e. in the segmentation map).
#           https://arxiv.org/pdf/1505.04597.pdf
#     - What we could do instead:
#       - In the encoder, connect also lower-level feature maps from previous CNN layers to the `Dense` layer that computes the latent representation.
#         Not sure if this will help, though, because we want the latent representation to encode high-level features, which live in the output of the
#         final CNN layer.
#       - In the decoder, add skips from the code point `z` to each layer; see the skip-VAE by Dieng et al. (2019):
#           https://arxiv.org/pdf/1807.04863.pdf
#         This adds a shorter path that promotes the use of different information (vs. going through the decoder ResNet blocks).
#         Model variant 11 does exactly this.

def make_codec(variant):
    """Set up compatible encoder and decoder neural networks for a CVAE.

    Return value is a `tuple` of two `tf.keras.Model`s::

        (encoder, decoder)

    Mainly meant for use by `CVAE.__init__`.

    We provide several variants of the CVAE, with different neural network geometries.
    See the source code for details.
    """
    # --------------------------------------------------------------------------------
    # Encoder

    def make_encoder():
        encoder_inputs = tf.keras.Input(shape=(28, 28, 1))

        # # Data augmentation on GPU (used automatically when the encoder is called with `training=True`)
        # # This is very slow, so let's not do it for now.
        # x = tf.keras.layers.RandomRotation(factor=(-0.05, 0.05),  # 5% = 18°
        #                                    fill_mode="constant", fill_value=0.0)(encoder_inputs)
        # x = tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
        #                                       fill_mode="constant", fill_value=0.0)(x)
        x = encoder_inputs

        if variant == 0:  # classical VAE
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                                       strides=2, padding="same")(x)     # 28×28×1 → 14×14×32
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                                       strides=2, padding="same",
                                       name="cnn_output")(x)             # 14×14×32 → 7×7×64

        elif variant == 1:  # ResNet attempt 1 (performs about as well as attempt 2)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                                       strides=2, padding="same")(x)                  # 28×28×1 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, bottleneck_factor=1)(x)    # 14×14×32→ 14×14×32
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                                       strides=2, padding="same")(x)                  # 14×14×32 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, bottleneck_factor=1,
                                name="cnn_output")(x)                                 # 7×7×64 → 7×7×64

        elif variant == 2:  # ResNet attempt 2 - large shallow model, good results
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu",
                                   bottleneck_factor=1)(x)               # 28×28×1 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                                bottleneck_factor=1)(x)                  # 14×14×32→ 14×14×32
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu",
                                   bottleneck_factor=1)(x)               # 14×14×32 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                bottleneck_factor=1,
                                name="cnn_output")(x)                    # 7×7×64 → 7×7×64

        elif variant == 3:  # ResNet attempt 3 - default bottleneck factor of 4, smaller model, but more blurred output
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu")(x)               # 28×28×1 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu")(x)               # 14×14×32 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                name="cnn_output")(x)                                             # 7×7×64 → 7×7×64

        elif variant == 4:  # ResNet attempt 4
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu")(x)               # 28×28×1 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu")(x)                  # 14×14×32→ 14×14×32
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu")(x)               # 14×14×32 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu")(x)                  # 7×7×64 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                name="cnn_output")(x)                                             # 7×7×64 → 7×7×64

        elif variant == 5:  # ResNet attempt 5
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu",
                                   bottleneck_factor=2)(x)               # 28×28×1 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                                bottleneck_factor=2)(x)                  # 14×14×32 → 14×14×32
            x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                                bottleneck_factor=2)(x)                  # 14×14×32 → 14×14×32
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu",
                                   bottleneck_factor=2)(x)               # 14×14×32 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                bottleneck_factor=2)(x)                  # 7×7×64 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                bottleneck_factor=2,
                                name="cnn_output")(x)                    # 7×7×64 → 7×7×64

        elif variant == 6:  # ResNet attempt 6 - deeper network (more layers) - very good results
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation="relu",
                                   bottleneck_factor=2)(x)                   # 28×28×1 → 14×14×32
            for _ in range(3):
                x = IdentityBlock2D(filters=32, kernel_size=3, activation="relu",
                                    bottleneck_factor=2)(x)                  # 14×14×32 → 14×14×32
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation="relu",
                                   bottleneck_factor=2)(x)               # 14×14×32 → 7×7×64
            for _ in range(2):
                x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                    bottleneck_factor=2)(x)                  # 7×7×64 → 7×7×64
            x = IdentityBlock2D(filters=64, kernel_size=3, activation="relu",
                                bottleneck_factor=2,
                                name="cnn_output")(x)  # like previous two, but only the final CNN block should have this name.

        elif variant == 7:  # ResNet attempt 7 - wider network (more channels), 959 348 parameters, 4.4GB total VRAM usage (during training, for complete CVAE)
            # According to He et al. (2015), adding depth to a convolution network beyond a certain
            # (problem-dependent) point, accuracy starts to degrade. Instead, adding width (number of
            # channels, i.e. `filters`) can still increase the capacity of the model usefully.
            #   https://arxiv.org/abs/1502.01852
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = ProjectionBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = ConvolutionBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = ProjectionBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2,
                                name="cnn_output")(x)

        elif variant == 8:  # Dropout experiment - dropout after each spatial level (14×14, 7×7)
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = ProjectionBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)

            x = ConvolutionBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = ProjectionBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = IdentityBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction,
                                        name="cnn_output")(x)

        elif variant == 9:  # Dropout experiment 2 - dropout after each ResNet block; best results up to this point (test ELBO 1360)
            x = ConvolutionBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = ProjectionBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)

            x = ConvolutionBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = ProjectionBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction,
                                        name="cnn_output")(x)

        elif variant in (10, 11):  # add another level of feature maps; test ELBO 1362
            # Level 1 - full spatial resolution, low-level feature map (28×28). (New.)
            #
            # A full bottleneck block is inefficient here at the input side, because we have only one
            # input channel, so the only thing the initial projection can do is to create copies of the
            # same data (introducing 28×28×8 = 6272 mixing coefficients that have no effect on the output).
            #
            # So to slightly optimize performance, let's use two-thirds of a bottleneck block,
            # skipping the initial channel mixer.
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                       kernel_initializer="he_normal",
                                       padding="same")(x)
            x = tf.keras.layers.PReLU()(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=1,
                                       kernel_initializer="he_normal",
                                       padding="same")(x)
            x = tf.keras.layers.PReLU()(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            # x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)  # for some reason, normalizing here breaks the whole NN (will not train usefully)

            # Then proceed as usual in a ResNet - remix the detected features, at the same number of channels.
            x = IdentityBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)  # output of level 1

            # Level 2 - spatial downscale and remix, generate mid-level feature map (14×14)
            x = ConvolutionBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)  # output of level 2

            # Level 3 - spatial downscale and remix, generate high-level feature map (7×7)
            x = ConvolutionBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)  # output of level 3

            # Level 4
            # Can't spatially downscale 7×7 any more (reversibly), so we just remix into a higher-dimensional
            # space (more channels) at the same spatial resolution. We already have a lot of channels to use
            # as input, so we hope also this part of the network will learn something useful. :)
            x = ProjectionBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                  bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlock2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction,
                                        name="cnn_output")(x)  # final CNN output

        else:
            raise ValueError(f"Unknown model variant {variant}, see source code for available models")

        x = tf.keras.layers.Flatten()(x)

        # TODO: How well does this work without the extra Dense layer? Better; let's drop it!
        #
        # # VRAM saving trick from the Keras example: the encoder has *two* outputs: mean and logvar. Hence,
        # # if we add a small dense layer after the convolutions, and connect that to the outputs, we will have
        # # much fewer trainable parameters (in total) than if we connect the last convolution layer to the outputs
        # # directly. As pointed out by:
        # # https://linux-blog.anracom.com/2022/10/23/variational-autoencoder-with-tensorflow-2-8-xii-save-some-vram-by-an-extra-dense-layer-in-the-encoder/
        # #
        # # Note that this trick only helps if `2 * latent_dim > extra_layer_size`; otherwise the architecture
        # # with the extra layer uses *more* VRAM. However, in that scenario it increases the model capacity slightly
        # # (but in practice we get better results if we add more layers in the convolution part instead of making
        # # this one larger).
        # x = tf.keras.layers.Dense(units=extra_layer_size)(x)
        # x = tf.keras.layers.PReLU()(x)

        # No activation function in the output layers - we want arbitrary real numbers as output.
        # The outputs will be interpreted as `(μ, log σ)` for the variational posterior qϕ(z|x).
        # A uniform distribution for these quantities (from the random initialization of the NN)
        # is a good prior for unknown location and scale parameters, see e.g.:
        #   https://en.wikipedia.org/wiki/Principle_of_transformation_groups
        #   https://en.wikipedia.org/wiki/Principle_of_maximum_entropy
        z_mean = tf.keras.layers.Dense(units=latent_dim, name="z_mean", dtype="float32")(x)
        z_log_var = tf.keras.layers.Dense(units=latent_dim, name="z_log_var", dtype="float32")(x)
        encoder_outputs = [z_mean, z_log_var]

        return tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")

    # --------------------------------------------------------------------------------
    # Decoder

    def make_decoder():
        decoder_inputs = tf.keras.Input(shape=(latent_dim,))

        # How well does this work without the extra Dense layer? Better; let's drop it!
        #
        # # Here we add the dense extra layer just for architectural symmetry with the encoder.
        # x = tf.keras.layers.Dense(units=extra_layer_size)(decoder_inputs)
        # x = tf.keras.layers.PReLU()(x)
        x = decoder_inputs

        # The size of the final encoder convolution output depends on the model variant.
        encoder_cnn_output_layer = encoder.get_layer(name="cnn_output")  # `encoder` initialized below before we call `make_decoder`
        encoder_cnn_final_shape = encoder_cnn_output_layer.output_shape[1:]  # ignore batch dimension
        x = tf.keras.layers.Dense(units=prod(encoder_cnn_final_shape))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Reshape(target_shape=encoder_cnn_final_shape)(x)

        # Now we are at the point of the architecture where we have the Conv2D output,
        # so let's mirror the encoder architecture to return to the input space.
        #
        # Note no activation function in the output layer - we want arbitrary real numbers as output.
        # The output will be interpreted as parameters `P` for the observation model pθ(x|z).
        # Here we want just something convenient that we can remap as necessary.

        if variant == 0:  # Classical VAE
            x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
                                                strides=2, padding="same")(x)     # 7×7×64 → 14×14×32
            x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2,
                                                padding="same")(x)                # 14×14×32 → 28×28×1

        elif variant == 1:  # ResNet attempt 1 (performs about as well as attempt 2)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, bottleneck_factor=1)(x)    # 7×7×64 → 7×7×64
            x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu",
                                                strides=2, padding="same")(x)                  # 7×7×64 → 14×14×32
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, bottleneck_factor=1)(x)    # 14×14×32 → 14×14×32
            x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3,
                                                strides=2, padding="same")(x)                  # 14×14×32 → 28×28×1

        elif variant == 2:  # ResNet attempt 2 - large shallow model, good results
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu",
                                         bottleneck_factor=1)(x)                   # 7×7×64 → 7×7×64
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                            bottleneck_factor=1)(x)                # 7×7×64 → 14×14×32
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                         bottleneck_factor=1)(x)                   # 14×14×32 → 14×14×32
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3,
                                            bottleneck_factor=1)(x)                # 14×14×32 → 28×28×1

        elif variant == 3:  # ResNet attempt 3 - default bottleneck factor of 4, smaller model, but more blurred output
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)  # 7×7×64 → 14×14×32
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)                      # 14×14×32 → 28×28×1

        elif variant == 4:  # ResNet attempt 4
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu")(x)     # 7×7×64 → 7×7×64
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)  # 7×7×64 → 14×14×32
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu")(x)     # 14×14×32 → 14×14×32
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)                      # 14×14×32 → 28×28×1

        elif variant == 5:  # ResNet attempt 5
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
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)  # 14×14×32 → 28×28×1

        elif variant == 6:  # ResNet attempt 6 - deeper network (more layers) - very good results
            for _ in range(3):
                x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation="relu",
                                             bottleneck_factor=2)(x)     # 7×7×64 → 7×7×64
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                            bottleneck_factor=2)(x)  # 7×7×64 → 14×14×32
            for _ in range(3):
                x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation="relu",
                                             bottleneck_factor=2)(x)     # 14×14×32 → 14×14×32
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3)(x)  # 14×14×32 → 28×28×1

        elif variant == 7:  # ResNet attempt 7 - wider network (more channels), 755 810 parameters, 4.4GB total VRAM usage (during training, for complete CVAE)
            x = IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = IdentityBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ConvolutionBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                            bottleneck_factor=2)(x)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ProjectionBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3,
                                            bottleneck_factor=2)(x)

        elif variant == 8:  # Dropout experiment - dropout after each spatial level (14×14, 7×7)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = IdentityBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ConvolutionBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                            bottleneck_factor=2)(x)

            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ProjectionBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3,
                                            bottleneck_factor=2)(x)

        elif variant == 9:  # Dropout experiment 2 - dropout after each ResNet block; best results up to this point (test ELBO 1360)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                            bottleneck_factor=2)(x)

            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = ProjectionBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=1, kernel_size=3,
                                            bottleneck_factor=2)(x)

        elif variant == 10:
            # Level 4
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)

            # Level 3
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=64, kernel_size=3,
                                            bottleneck_factor=2)(x)

            # Level 2
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3,
                                            bottleneck_factor=2)(x)

            # Level 1
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)

            # The inverse of the final two-thirds of a bottleneck block:
            # x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=1,
                                                kernel_initializer="he_normal",
                                                padding="same")(x)
            x = tf.keras.layers.PReLU()(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
                                                kernel_initializer="he_normal",
                                                padding="same")(x)
            x = tf.keras.layers.PReLU()(x)

        elif variant == 11:  # Otherwise like 10, but with extra skip-connections. Test ELBO 1362.
            # Inspired by skip-VAE of Dieng et al. (2019); connect the code point `z` to the input of each decoder layer.
            #     https://arxiv.org/pdf/1807.04863.pdf
            # As to how to connect it when we don't use a conditional decoder that can accept concatenated inputs
            # (like the Gated PixelCNN of van den Oord et al., used in the original study), see the code by Dieng et al.:
            #     https://github.com/yoonkim/skip-vae/blob/master/models_img_skip.py
            # Especially, see the `MLPVAE`, which just projects by a `torch.nn.Linear` layer (the Keras equivalent
            # of which is a `tf.keras.layers.Dense` with default settings), and then adds the result to `x`,
            # much like a residual connection.
            #
            # So we're essentially just setting up classical residual connections. However, by connecting them
            # directly to the decoder inputs (code point `z`), this path is shorter, and thus promotes the use
            # of different information than the residual path through the ResNet blocks.
            def connect_input(x):
                # HACK, because no `.numpy()` method available at this stage
                x_shape = tf.shape(x)._inferred_value[1:]  # ignore batch dimension
                skip = tf.keras.layers.Dense(units=prod(x_shape))(decoder_inputs)
                skip = tf.keras.layers.Reshape(target_shape=x_shape)(skip)
                return tf.keras.layers.Add()([x, skip])

            # Level 4
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=256, rate=dropout_fraction)(x)
            x = ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                           bottleneck_factor=2)(x)

            # Level 3
            x = connect_input(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=128, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=64, kernel_size=3,
                                            bottleneck_factor=2)(x)

            # Level 2
            x = connect_input(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)
            x = GNDropoutRegularization(groups=64, rate=dropout_fraction)(x)
            x = ConvolutionBlockTranspose2D(filters=32, kernel_size=3,
                                            bottleneck_factor=2)(x)

            # Level 1
            x = connect_input(x)
            x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = IdentityBlockTranspose2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                         bottleneck_factor=2)(x)

            # The inverse of the final two-thirds of a bottleneck block:
            # x = GNDropoutRegularization(groups=32, rate=dropout_fraction)(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=1,
                                                kernel_initializer="he_normal",
                                                padding="same")(x)
            x = tf.keras.layers.PReLU()(x)
            x = tf.keras.layers.SpatialDropout2D(rate=dropout_fraction)(x)
            x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1,
                                                kernel_initializer="he_normal",
                                                padding="same")(x)
            x = tf.keras.layers.PReLU()(x)

        else:
            raise ValueError(f"Unknown model variant {variant}, see source code for available models")

        # Cast final output of decoder to float32. Important when running under a mixed-precision policy.
        # We do this also for float32 (it's then a no-op), so that the topology of the NN does not depend on the policy.
        #
        # Should use slightly less VRAM, if done here as a separate operation (need at most 2 float32s per pixel)
        # instead of using float32 as the dtype of the final convolution transpose block (which does actual compute).
        #   https://tensorflow.org/guide/mixed_precision
        decoder_outputs = tf.keras.layers.Activation("linear", dtype="float32")(x)  # identity function, cast only

        return tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    encoder = make_encoder()
    decoder = make_decoder()
    return encoder, decoder


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder.

    The optimizer should be configured as `model.compile(optimizer=...)`.

    This class defines its metrics and losses manually; those should
    **not** be configured in `compile`.

    Encoding takes pixel data `x` → parameters of variational posterior qϕ(z|x), namely `(μ, log σ)`.
    Reparameterization takes those parameters and yields a code point `z`. The code point is chosen
    stochastically, by drawing one sample from qϕ(z|x).

    Decoding takes a code point `z` → parameters of observation model pθ(x|z), one value per pixel.
    Applying a sigmoid to this yields the λ parameter of the continuous Bernoulli distribution,
    which can be used as the pixel value as-is.

    To encode x → z::

        mean, logvar = model.encoder(x, training=False)  # compute code distribution parameters for given `x`
        eps, z = model.reparameterize(mean, logvar)  # draw a single sample from the code distribution

    To decode z → xhat::

        xhat = model.sample(z, training=False)
    """

    def __init__(self, *, latent_dim=2, variant=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.variant = variant
        self.encoder, self.decoder = make_codec(variant)
        # https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        #     https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        return [self.loss_tracker]

    # # We shouldn't need this, because we define `metrics`.
    # def reset_metrics(self):
    #     self.loss_tracker.reset_states()

    # --------------------------------------------------------------------------------
    # Save/load support.
    #   https://www.tensorflow.org/guide/keras/serialization_and_saving
    #   https://www.tensorflow.org/tutorials/keras/save_and_load
    #
    # To make a custom `Model` saveable in TF 2.12 or later, it must:
    #   - Provide a `get_config` instance method, which returns a JSON-serializable dictionary of parameters
    #     that were used to create this instance.
    #   - Provide a `from_config` class method, which takes such a dictionary and instantiates the model.
    #     Note it's enough that the constructor sets up the correct NN structure when instantiated with these parameters;
    #     Keras handles the actual loading of the coefficient data into that NN structure.
    #   - Preferably, the class should register itself: `@tf.keras.utils.register_keras_serializable(package="MyPackage")`
    # ...but it only needs to do that if the constructor takes any args that are NOT:
    #   - bare basic Python types (str, int, ...)
    #   - Keras objects, which already know how to serialize
    # If the ctor only takes those types, then there is no need to override anything.
    #
    #   - I'm not sure if this is the case anymore, but it used to be (before TF 2.12) that a custom model must also provide a
    #     `call` method to be saveable. For an autoencoder, such a method is basically useless, except as documentation for
    #     how to make a full round-trip through the AE.
    #   - In any case, any code that wishes to call `save` should first force the model to build its graph, with something like::
    #       model.build((batch_size, 28, 28, 1))
    #     or alternatively::
    #       dummy_data = tf.random.uniform((batch_size, 28, 28, 1))
    #       _ = model(dummy_data)
    #
    def get_config(self):
        config = super().get_config()
        # The `variant` parameter specifies which actual NN structure is instantiated by our constructor.
        # Once a given `variant` value has been used in a version pushed to GitHub, it should be treated as part of the public API.
        # That is, for backward compatibility, it should forever refer to that NN structure, so that future versions can load old checkpoints.
        config.update({"latent_dim": self.latent_dim,
                       "variant": self.variant})
        return config
    # # Default implementation - here for reference only. We can use it as-is.
    # @classmethod
    # def from_config(cls, config):
    #     model = cls(**config)
    #     return model
    @tf.function
    def call(self, x, training=None):
        """Send data batch `x` on a full round-trip through the autoencoder. (Included for API compatibility only.)"""
        # See `elbo_loss` for a detailed explanation of the internals.
        # encode to latent representation
        mean, logvar = self.encoder(x, training=training)  # compute code distribution parameters for given `x`
        eps, z = self.reparameterize(mean, logvar)  # draw a single sample from the code distribution
        # decode from latent representation
        xhat = self.sample(z, training=training)
        return xhat

    # --------------------------------------------------------------------------------
    # Custom training and testing.
    #   https://keras.io/getting_started/faq/#what-if-i-need-to-customize-what-fit-does
    #
    # Basically (TODO: AFAIK; check, and fix if wrong):
    #   - Overriding `train_step` customizes what `fit` does
    #   - Overriding `test_step` customizes what `evaluate` does
    #   - Overriding `call` customizes what `predict` does, but:
    #     - if no custom `train_step`, also affects `fit` (which calls the model with `training=True`)
    #     - if no custom `test_step`, also affects `evaluate` (which calls the model with `training=False`)

    @tf.function
    def train_step(self, x):
        """Execute one training step, computing and applying gradients via backpropagation.

        Supports fp32, fp16 and bf16 precision.

        `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures
        """
        policy = tf.keras.mixed_precision.global_policy()
        fp16 = (policy.compute_dtype == "float16")  # fp16 needs loss scaling, fp32 and bf16 do not

        with tf.GradientTape() as tape:
            loss = elbo_loss(self, x, training=True)  # TODO: maybe should be a method?
            if fp16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)  # mixed precision

        # Compute gradients
        trainable_vars = self.trainable_variables
        if fp16:
            scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, x):
        """Execute one testing step.

        Note we have no separate target data `y`, because this is an autoencoder.

        `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures
        """
        self.compute_loss(x)
        return {m.name: m.result() for m in self.metrics}

    # TODO: is this method correct?
    @tf.function
    def compute_loss(self, x):
        """Compute the total loss for a batch, at inference time.

        Provided for API compatibility.
        """
        loss = elbo_loss(self, x, training=False)
        self.loss_tracker.update_state(loss)
        return loss

    # --------------------------------------------------------------------------------
    # Other custom methods, specific to a CVAE.

    @tf.function
    def reparameterize(self, mean, logvar):
        """Map code distribution parameters to a code point stochastically.

        (μ, log σ) → z

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
        # The noise sample is drawn from a unit spherical Gaussian N(μ=0, σ=1), distinct from the latent prior.
        #
        # eps = tf.random.normal(shape=mean.shape)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        return eps, z

    @tf.function
    def sample(self, z=None, training=None):
        """Sample from the observation model pθ(x|z), returning the pixel-wise means.

        `z`: tf array of size `(batch_size, latent_dim)`.
             If not specified, a batch of 100 random samples is returned.
        """
        if z is None:
            # Sample the code points from the latent prior.
            z = tf.random.normal(shape=(100, self.latent_dim))
        P = self.decoder(z, training=training)
        lam = tf.sigmoid(P)  # probability for discrete Bernoulli; λ parameter of continuous Bernoulli
        # For the discrete Bernoulli distribution, the mean is the same as the Bernoulli parameter,
        # which is the same as the probability of the output taking on the value 1 (instead of 0).
        # For the continuous Bernoulli distribution, we just return λ as-is (works fine as output in practice).
        return lam

    # --------------------------------------------------------------------------------
    # Deprecated methods

    def my_save(self, path: str):
        """Legacy custom saver, saving the encoder and decoder Functional models separately.

        *DEPRECATED*: Use `model.save("my_snapshot.keras", save_format="keras_v3")` instead.
        """
        clear_and_create_directory(path)
        p = pathlib.Path(path).expanduser().resolve()
        self.encoder.save(str(p / "encoder"))
        self.decoder.save(str(p / "decoder"))
    def my_load(self, path: str):
        """Legacy custom loader, loading the encoder and decoder Functional models separately.

        *DEPRECATED*: Use `model = tf.keras.models.load_model("my_snapshot.keras")` instead.
        """
        p = pathlib.Path(path).expanduser().resolve()
        self.encoder = tf.keras.models.load_model(str(p / "encoder"))
        self.decoder = tf.keras.models.load_model(str(p / "decoder"))

# --------------------------------------------------------------------------------
# Loss function (and its helpers)

# Where the `log_normal_pdf` formula comes from: recall the PDF of the normal distribution:
#   N(x; μ, σ) := (1 / (σ √(2π))) exp( -(1/2) * ((x - μ) / σ)² )
# where
#   mean := μ,
#   var := σ²,  logvar := log σ²
#
# We have
#   N(x; μ, σ)  = (1 / (exp(log σ) exp(log √(2π)))) exp( -(1/2) * ((x - μ) / σ)² )    [a = exp(log(a))]
#               = exp(-log σ) exp(-log √(2π)) exp( -(1/2) * ((x - μ) / σ)² )          [1 / exp(a) = exp(a)^(-1) = exp(-a)]
#               = exp( -log σ - log(√(2π)) - (1/2) * ((x - μ) / σ)² )                 [exp(a) exp(b) = exp(a + b)]
#               = exp( -log σ - (1/2) log(2π) - (1/2) * ((x - μ) / σ)² )              [log(a**b) = b log(a)]
#               = exp( -log σ - (1/2) log(2π) - (1/2) * (x - μ)² / σ² )
#               = exp( -log σ - (1/2) log(2π) - (1/2) * (x - μ)² / exp(log σ²) )
#               = exp( -log σ - (1/2) log(2π) - (1/2) * (x - μ)² * exp(-log σ²) )
#               = exp( -log σ - (1/2) log(2π) - (1/2) * (x - μ)² * exp(-logvar) )
#               = exp( -log(√(σ²)) - (1/2) log(2π) - (1/2) * (x - μ)² * exp(-logvar) )
#               = exp( -(1/2) log(σ²) - (1/2) log(2π) - (1/2) * (x - μ)² * exp(-logvar) )
#               = exp( -(1/2) logvar - (1/2) log(2π) - (1/2) * (x - μ)² * exp(-logvar) )
#               = exp( -(1/2) [logvar + log(2π) + (x - μ)² * exp(-logvar)] )
#
# Therefore,
#   log N(x; μ, σ) = -(1/2) [logvar + log(2π) + (x - μ)² * exp(-logvar)]
# as claimed.
#
# Note that since we have defined the reparameterization as
#   z = mean + eps * exp(logvar / 2)
# inverting yields
#   eps = (z - mean) * exp(-logvar / 2)
# and
#   eps² = (z - mean)² * exp(-logvar)
# so calling log_normal_pdf(z, mean, logvar) actually yields
#   sum_i(-0.5 * (eps_i**2 + logvar + log2pi))
# which matches Kingma and Welling (2019, algorithm 2).
#
# Note this is a multivariate gaussian with diagonal covariance, and we sum the logs,
# hence we take the product of the marginal probability densities of each component,
# thus yielding the probability at point `x` (just one real number... for each input in the batch).
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
    # cut_lam below might appear useless, but it is important to not evaluate log_norm near 0.5
    # as tf.where evaluates both options, regardless of the value of the condition.
    cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tf.math.log(tf.abs(2.0 * tf.atanh(1 - 2.0 * cut_lam))) - tf.math.log(tf.abs(1 - 2.0 * cut_lam))
    taylor = tf.math.log(2.0) + 4.0 / 3.0 * tf.pow(lam - 0.5, 2) + 104.0 / 45.0 * tf.pow(lam - 0.5, 4)
    return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), log_norm, taylor)

@tf.function
def log_px_z(model, x, z, training=None):
    """Compute `log pθ(x|z)`.

    Note this is a probability density, which can be larger than unity,
    so the logarithm may be positive (and large).

    `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures
    `z`: tensor of shape (N, latent_dim); data batch of code points
         (drawn from variational posterior; get them by encoding `x`
          and reparameterizing)
    """
    # Many tutorial implementations are unnecessarily confusing here; logits and the binary cross-entropy
    # are a sideshow, specific to the discrete Bernoulli observation model of the classic VAE, which
    # doesn't even make sense for continuous (grayscale) data. Hence many tutorials unnecessarily
    # binarize the input data to make it fit the discrete observation model, which hurts quality.
    # (Not to mention that then, if as is customary, the decoder returns the pixelwise means as the
    # decoded output, the resulting picture is not even a valid sample from the chosen observation model.)
    #
    # It is correct that even in the general case, we do want to minimize cross-entropy, but the ELBO is
    # easier to understand without introducing this extra concept.
    #
    # See e.g. Wikipedia on cross-entropy:
    #    https://en.wikipedia.org/wiki/Cross_entropy
    #
    # How to compute pθ(x|z) in general, for any VAE: take the observation parameters P computed by
    # P = NN_dec(θ, z) at the sampled z (thus accounting for the dependence on z), and evaluate the
    # known function pθ(x|z) (parameterized by P and x) with those parameters P at the input x.
    #
    # Note that unlike in a classical AE, in a VAE the decoded output is not directly x-hat (i.e. an
    # approximation of the input data point), but parameters for a distribution that can be used to
    # compute x-hat. Strictly, in a VAE, x-hat is a (pixelwise) distribution. Many VAE implementations
    # return the pixelwise means of that distribution as the decoded picture.
    #
    # For more details, see the VAE tutorial paper by Kingma and Welling (2019, algorithm 2).
    P = model.decoder(z, training=training)

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
    # when x is continuous in the interval [0, 1] (instead of discrete in the set {0, 1}).
    #
    # Note the output is now the parameter λ of the continuous Bernoulli distribution, not directly a
    # probability; and the mean of the continuous Bernoulli distribution is also different from λ.
    # Still, in practice λ works well as-is as a deterministic output value for the decoder.
    #
    # As for how to apply the normalization constant, see the original implementation by Loaiza-Ganem
    # and Cunningham: https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/cb_vae_mnist.ipynb
    lam = tf.sigmoid(P)  # interpret decoder output as logits; map into λ parameter of continuous Bernoulli

    # log PDF of continuous Bernoulli
    lam = tf.clip_by_value(lam, 1e-4, 1 - 1e-4)  # avoid log(0)
    logpx_z = x * tf.math.log(lam) + (1 - x) * tf.math.log(1 - lam) + cont_bern_log_norm(lam)  # log pθ(x|z) (observation model)

    # multivariate (pixels of `x`)
    # sum of logs = log of product of pixelwise probability densities
    #             = joint log-probability for independently distributed pixels
    return tf.reduce_sum(logpx_z, axis=[1, 2, 3])  # log pθ(x|z) (observation model)


@tf.function
def elbo_loss(model, x, training=None):
    """VAE loss function: negative of the ELBO, for a data batch `x`.

    Evaluated by drawing a single-sample Monte Carlo estimate. Kingma and Welling (2019) note
    that this yields an unbiased estimator for the expectation that appears in the ELBO formula.

    Note that we can't use any expensive methods here, because this runs in the inner loop of the
    NN optimization.
    """
    # Without explanation this calculation is overly magic, so let's comment step by step.

    # Encode the input data (pixels) into parameters for the variational posterior qϕ(z|x):
    #   x → NN_enc(ϕ, x) = (μ(ϕ, x), log σ(ϕ, x))
    # Here ϕ are the encoder NN coefficients.
    #
    # Note the output dimensionality; the encoder converts from the raw data space to the latent space.
    # Both `mean` and `logvar` have `latent_dim` components, so we output `2 * latent_dim` components.
    #
    # We must emphasize that unlike in the classical AE, in a VAE the encoder output is not a single encoded
    # data point z, but rather, parameters for a distribution qϕ(z|x), which for the given x, represents
    # the distribution of such points z.
    mean, logvar = model.encoder(x, training=training)

    # We could use multiple-sample Monte Carlo; see `negative_log_likelihood` for how to average the samples
    # correctly (we're dealing with log-probabilities, and expectation doesn't commute with log).
    #
    # Using more than one sample multiplies the computational cost, but usually increases quality only slightly.
    # At least qualitatively, it seems it is more cost-efficient (for test-set ELBO gain per unit of wall time
    # spent in training) to just use a larger dataset instead of improving the sampling here.
    #
    # Note that the expectation of a normally distributed variable, such as the latent point z, is just its
    # mean. So why do we need `logvar` at all - why don't we just use `mean` as an exact estimate of the
    # expectation of z ~ qϕ(z|x), since in the ELBO, all we actually want is an expectation?
    #
    # An important reason is that we are not just evaluating `z`; we must evaluate all terms of the ELBO consistently.
    # A Monte Carlo point estimate is a computationally efficient way to achieve this consistency.

    # Sample the variational posterior qϕ(z|x) to obtain *a* latent point z corresponding to the input x.
    # That is, draw a single sample z ~ qϕ(z|x), using a single noise sample ε ~ p(ε) and the deterministic
    # reparameterization transformation z = g(ε, ϕ, x).
    #
    # In the implementation, actually z = g(μ, log σ); the dependencies on ϕ and x have been absorbed into
    # μ(ϕ, x) and log σ(ϕ, x). The `reparameterize` function internally draws the noise sample ε, so we
    # don't need to supply it here.
    #
    # We choose our class of variational posteriors (which we optimize over) as factorized Gaussian,
    # mainly for convenience. Thus we interpret the (initially arbitrary) numbers coming from the encoder
    # as (μ, log σ) for a factorized Gaussian, and plug them in in those roles.
    #
    # Note that initially, the encoder output is essentially just `2 * latent_dim` arbitrary real numbers.
    # The encoder itself does not know the meaning of its outputs. That meaning is established *here*,
    # by the training process. *Because* we use the encoder outputs as (μ, log σ), the training process
    # adjusts the encoder coefficients so that those numbers actually become to represent (i.e. converge
    # to reasonable values for) (μ, log σ).
    #
    # (Let that sink in for a moment: the training process guides some arbitrary outputs to "magically"
    # become what we wanted them to represent. As always, the magic is an illusion; essentially, this
    # setup is no different from a classical optimization problem, say, over polynomials. We fix the
    # structure of a computational algorithm (which is now much more complex than a single polynomial)
    # and then optimize to find good values for its coefficients. The structure of the algorithm
    # fixes the meaning of the coefficients. Only humans see the meaning; the optimizer sees just
    # a vector of coefficients, which it tunes in order to improve the value of the objective function.)
    #
    # Finally, we re-emphasize that z is encoded stochastically. Even feeding in the same x produces
    # a different z each time (for each MC sample), since z is sampled from the variational posterior.
    eps, z = model.reparameterize(mean, logvar)

    # Decode the sampled `z`, obtain parameters (at each pixel) for observation model pθ(x|z):
    #   z → NN_dec(θ, z) = P(θ, z)
    # Here θ are the decoder NN coefficients. In our implementation, the observation model
    # takes just one parameter (per pixel), P.
    #
    # Note the output dimensionality: the decoder maps a latent point z (a point, not a distribution!)
    # back to the raw data space (pixel space). Since we work with grayscale pictures, the output has
    # one P value per pixel.
    #
    # We implement the classic VAE: we choose our class of observation models as factorized Bernoulli
    # (pixel-wise). Thus we interpret the (again, initially arbitrary) numbers coming from the decoder
    # as logits for a factorized Bernoulli distribution, and plug them in in that role.
    #
    logpx_z = log_px_z(model, x, z, training=training)

    # We choose the latent prior pθ(z) to be a spherical unit Gaussian, N(μ=0, σ=1).
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
    # (here the right-hand side) is an unbiased estimator of the other side. The definition
    # on the first line is sometimes called the *joint-contrastive* form of the ELBO.
    #
    # The last line above is the one that is easily computable. For understanding how the ELBO works,
    # consider an alternative form:
    #
    #   ELBO[θ,ϕ](x) ≡ E[qϕ(z|x)] [log pθ(x,z) - log qϕ(z|x)]
    #                = E[qϕ(z|x)] [log (pθ(x|z) pθ(z)) - log qϕ(z|x)]
    #                = E[qϕ(z|x)] [log pθ(x|z) - [log qϕ(z|x) - log pθ(z)]]
    #                ≡ E[qϕ(z|x)] [log pθ(x|z)] - DKL(qϕ(z|x) ‖ pθ(z))
    #
    # where DKL is the Kullback--Leibler divergence. So the above expression is the sum of
    # a reconstruction quality term (expectation, with z drawn from qϕ(z|x), of the log-likelihood
    # of the input x, given the code z) and a regularization term (KL divergence of qϕ(z|x) from the
    # latent *prior*). This is the *prior-contrastive* form of the ELBO.
    #
    # On the other hand, by rewriting the joint probability the other way, we have also
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
    # approximate variational posterior qϕ(z|x) is to the unknown true *posterior* pθ(z|x).
    # So maximizing the ELBO over distributions qϕ(z|x), we should find the optimal qϕ(z|x),
    # which approximates pθ(z|x) best, within our chosen class of distributions qϕ(z|x).
    #
    # Finally, keep in mind that for a continuous distribution p(x), the ELBO may actually take on
    # positive values, because p(x) is then a probability *density*, which may in general exceed 1.
    elbo = tf.reduce_mean(logpx_z + logpz - logqz_x)

    return -elbo  # with sign flipped → ELBO loss


# --------------------------------------------------------------------------------
# Performance statistics

# TODO: These should perhaps be moved into a separate module (`performance_statistics.py`?),
# TODO: since there is a lot of wordy math documentation here.

def active_units(model, x, *, batch_size=1024, eps=0.1):
    """[performance statistic] Compute AU, the number of latent active units.

    `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures

    Returns an int, the number of latent active units for given model,
    estimated using the given data.

    It is preferable to pass as much data as possible (e.g. all of the
    test data) to get a good estimate of AU.

    AU measures how many of the available latent dimensions a trained VAE actually uses,
    so the maximum possible value is `latent_dim`. A higher value is better.

    The log of the covariance typically has a bimodal distribution, so AU is not very
    sensitive to the value of ϵ, as long as it is between the peaks.

    We define AU as::

        AU := #{i = 1, ..., d: abs(cov_x( E[z ~ qϕ(z|x)](zi) )) > ϵ}

    where d is the dimension of the latent space, ϵ is a suitable small number,
    and #{} counts the number of elements of a set.

    See Burda et al. (2016):
       https://arxiv.org/abs/1509.00519
    """
    # As a classical numericist totally not versed in any kind of statistics / #helpwanted:
    #
    # Burda's original definition of the activation statistic AU is essentially
    #
    #   AU := #{u: Au > ϵ}
    #
    # where (and I quote literally)
    #
    #   Au := Cov_{x}( E(u ∼ qφ(u|x))[u] )
    #
    # Here u is a component of the code point vector z (in our notation, u → zi),
    # and the x is boldface, denoting the whole input picture. The authors used ϵ = 0.01.
    #
    # No one else (except other papers citing this exact definition from Burda et al.)
    # seems to use that subscript notation for covariance, not to mention using an initial
    # capital "C"; I couldn't find a definitive definition *anywhere* pinning down the
    # exact meaning of this variant of the notation.
    #
    # The text in section 5.2 of the paper implies it is indeed some kind of covariance,
    # since the use case of this activation statistic is to measure whether each u (i.e. zi)
    # affects the output of the generative model (decoder) or not.
    #
    # Two more details are missing from the paper:
    #
    #   - The input picture x, interpreted as a data vector, has  n_pixels = ny * nx * c
    #     components. The latent code z has latent_dim components. Therefore,
    #     the sample covariance between observations x and z is a matrix of size
    #     [n_pixels, latent_dim]. But the paper hints that the maximum possible value
    #     of AU is latent_dim; which implies we should reduce over the pixels, leaving
    #     only latent_dim components for the covariance (aggregate effect of each zi on
    #     the picture x). Should we sum over the pixels? Take the mean over the pixels?
    #     Something else? (We have chosen to sum.)
    #
    #   - Generally, covariance may also be negative, the sign giving the sense of
    #     the detected linear relationship. Yet in Appendix C, the authors speak of
    #     plotting its (real-valued) log, which for a negative input is clearly NaN.
    #     I think the definition must be missing an abs(), unless this is implied by
    #     the notation "Cov_{x}(...)" instead of the standard "cov(x, ...)".
    #     (We have chosen to take abs() before comparing to ϵ.)
    #
    # If I understand this right,  E[z ~ qϕ(z|x)](z) = μ  from the encoder.
    # We give the encoder the variational parameters ϕ (trained NN coefficients)
    # and an input picture x, and it gives us a multivariate gaussian with diagonal
    # covariance, parameterized by the vectors (μ, log σ), and conditioned on the
    # input x. For the given input x, the expectation of this gaussian is μ.
    #
    # To compute the covariance, we must then encode the whole dataset (that we wish
    # to use to estimate AU), and find the sample mean of this expectation, μbar.
    #
    # Covariance between two continuous random variables x and y is defined as
    #
    #   covar(x, y) := ∫ (x - xbar) (y - ybar) dp
    #                = ∫ (x - xbar) (y - ybar) p(x, y) dx dy
    #
    # But we're working with a dataset, so more directly relevant for us is the
    # sample covariance:
    #
    #   covar(x, y) := (1 / (N - 1)) ∑k (xk - xbar) (yk - ybar)
    #
    # where k indexes the observations. Note we essentially want to correlate the
    # behavior of the random variables X and Y across observations, so we need equally many
    # observations xk and yk. (Which we indeed have, since encoding one x produces one μ.)
    #
    # The -1 is Bessel's correction, accounting for the fact that the population mean
    # is unknown, so we use the sample mean, which is not independent of the samples.
    μ, ignored_logσ = model.encoder.predict(x, batch_size=batch_size)
    xbar = tf.reduce_mean(x, axis=0)  # pixelwise mean (over dataset)
    μbar = tf.reduce_mean(μ, axis=0)  # latent-dimension-wise mean (over dataset)

    # Like the scatter matrix in statistics, but summed over pixels and channels of `x`.
    @batched(batch_size)  # won't fit in VRAM on the full training dataset
    def scatter(x, μ):  # ([N, xy, nx, c], [N, xy, nx, c]) -> [N, latent_dim]
        xdiff = tf.reduce_sum((x - xbar), axis=[1, 2, 3])  # TODO: is this the right kind of reduction here?
        outs = []
        for d in range(latent_dim):  # covar(x, z_d)
            outs.append(xdiff * (μ[:, d] - μbar[d]))  # -> [batch_size]
        return tf.stack(outs, axis=-1)  # -> [batch_size, latent_dim]
    N = tf.shape(x)[0]
    sample_covar = (1. / (float(N) - 1.)) * tf.reduce_sum(scatter(x, μ), axis=0)
    return int(tf.reduce_sum(tf.where(tf.greater(tf.math.abs(sample_covar), eps), 1.0, 0.0)))


# TODO: refactor `lopsum` to `util` and move the explanations to this docstring.
def logsum(logxs):
    """`log(∑k x[k])`, computed in terms of `log x[k]`, using the smoothmax identity.

    `logxs`: rank-1 tensor, containing `[log x[0], log x[1], ...]`

    Returns `log(∑k x[k])`.

    This is computed in a numerically stable way; we actually never evaluate `x[k]`.
    This is useful when only the logarithms are available (e.g. to prevent overflow
    from large exponents).
    """
    # We can accumulate the log samples with the help of the smoothmax identity
    # for the logarithm of a sum.
    #
    # Discussion summarized from:
    #   https://cdsmithus.medium.com/the-logarithm-of-a-sum-69dd76199790
    #
    # For all `x, y > 0`, it holds that:
    #
    #   log(x + y) = log(x * (1 + y / x))
    #              = log x + log(1 + y / x)
    #              = log x + log(1 + exp(log(y / x)))
    #              = log x + log(1 + exp(log y - log x))
    #              ≡ log x + softplus(log y - log x)
    # where
    #
    #   softplus(x) ≡ log(1 + exp(x))
    #
    # Incidentally, let us define a C∞ continuous analog of the `max` function:
    #
    #   smoothmax(x, y) := x + softplus(y - x)
    #
    # Upon close inspection (proof omitted here), we see that most of the
    # usual properties of addition (commutativity, associativity, distributivity)
    # hold for `smoothmax`. The identity for the logarithm of a sum shortens into:
    #
    #   log(x + y) = smoothmax(log x, log y)
    #
    # or in other words, the `log` of a sum is a smoothed maximum of the
    # logs of the summands.
    #
    # The one to watch out for is the identity property. Since we assumed
    # `x, y > 0` to keep all arguments in the domain of (real-valued)
    # `log`, strictly speaking the smoothmax identity is not applicable
    # when `y = 0`. In the limit, though, we have:
    #
    #   lim[y → -∞] smoothmax(x, y) = x
    #
    # so we *can* say that:
    #
    #   lim[y → 0+] log(x + y) = lim[y → 0+] smoothmax(log x, log y)
    #                          = log x
    #
    # Also keep in mind `smoothmax` is not `max`, although it behaves
    # somewhat similarly for most values of its arguments. It differs
    # from `max` the most when `|x - y|` is small. The extreme case is:
    #
    #   smoothmax(x, x) = x + softplus(0)
    #                   = x + log(1 + exp(0))
    #                   = x + log(1 + 1)
    #                   = x + log(2)
    #
    # Finally, observe that:
    #
    #   x + log 1 = x
    #   x + log 2 = smoothmax(x, x)
    #   x + log 3 = smoothmax(x, smoothmax(x, x))
    #   x + log 4 = smoothmax(x, smoothmax(x, smoothmax(x, x)))
    #   ...
    #
    # Proof by induction, omitted. The original author writes:
    #   [This] resembles a sort of definition of addition of log-naturals
    #   as “repeated smoothmax of a number with itself”, in very much the
    #   same sense that multiplication by naturals can be defined as
    #   repeated addition of a number with itself, strengthening the
    #   notion that this operation is sort-of one order lower than addition.
    #
    # This perhaps looks clearer if we use some symbol, say `𝕄`,
    # as infix notation for `smoothmax`:
    #
    #   x + log 1 = x
    #   x + log 2 = x 𝕄 x
    #   x + log 3 = x 𝕄 (x 𝕄 x)
    #   x + log 4 = x 𝕄 (x 𝕄 (x 𝕄 x))
    #   ...
    #
    # and since `smoothmax` is associative, we can drop the parentheses:
    #
    #   x + log 1 = x
    #   x + log 2 = x 𝕄 x
    #   x + log 3 = x 𝕄 x 𝕄 x
    #   x + log 4 = x 𝕄 x 𝕄 x 𝕄 x
    #   ...
    #
    # which indeed looks similar to
    #
    #   x * 1 = x
    #   x * 2 = x + x
    #   x * 3 = x + x + x
    #   x * 4 = x + x + x + x
    #   ...
    #
    # In analogy with the `log` of a product:
    #
    #   log(x y) = log x + log y
    #
    # the `𝕄` notation also gives a pretty expression for the `log` of a sum:
    #
    #   log(x + y) = log x 𝕄 log y
    #
    # where
    #
    #   x 𝕄 y := x + ⟦y - x⟧+    (smoothmax)
    #   ⟦x⟧+ := log(1 + exp(x))   (softplus, notation in analogy with positive part "[x]+")

    # The benefit of the smoothmax identity is that it allows us to work with
    # logarithms only, except for the evaluation of the softplus. Whenever its
    # argument is small, the `exp` can be taken without numerical issues.
    #
    # Although there is a risk of catastrophic cancellation in `log y - log x`,
    # we still want a reasonable amount of cancellation, to keep the argument
    # to softplus small. So we sort the logratios before summing.
    #
    # Starting the summation from the *smallest* numbers should allow us to accumulate them
    # before we lose the mantissa bits to represent them due to the increasing exponent.
    # If this is really important, we could do it in pure Python, and `math.fsum` them.
    # But we have likely already lost more accuracy due to cancellation, so let's not bother
    # overengineering this part.
    logxs = tf.sort(logxs, axis=0, direction="ASCENDING")
    # # What we want to do:
    # from unpythonic import window
    # out = acc[0]  # log(x[0])
    # for prev, curr in window(2, logxs):  # log(x[k]) - log(x[k-1])
    #     out += tf.math.softplus(curr - prev)
    sp_diffs = tf.math.softplus(logxs[1:] - logxs[:-1])  # softplus(log(x[k]) - log(x[k-1]))
    return logxs[0] + tf.reduce_sum(sp_diffs)


def negative_log_likelihood(model, x, *, batch_size=1024, n_mc_samples=10):
    """[performance statistic] Compute the negative log-likelihood (NLL).

    `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures

    Returns a float, the mean NLL for the given data, using the given model.

    When `x` is held-out data, NLL measures generalization (smaller is better).
    For a single input sample `x`, the NLL is defined as::

        log pθ(x) = -log( E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ) )

    This expression is intractable, so we approximate it using Monte Carlo::

        log pθ(x) ≈ -log( (1/S) ∑s (pθ(x, z[s]) / qϕ(z[s]|x)) )

    where `S = n_mc_samples` and z[s] are the Monte Carlo samples of z ~ qϕ(z|x).
    The NLL is pretty much the ELBO as used in VAE training, but with some differences:

      - Multiple MC samples to improve accuracy.
      - The mean is computed over the whole dataset `x`, not over each batch.

    For numerical reasons, we accumulate the MC samples without evaluating
    `pθ(x, z[s])` directly, preferring to work on `log pθ(x, z[s])` instead.
    Consider that the joint probability can be rewritten as

        pθ(x, z) = pθ(x|z) pθ(z)

    Whereas pθ(z) and qϕ(z|x) are gaussians, with reasonable log-probabilities,
    for the continuous-Bernoulli VAE, on MNIST, `log pθ(x|z) ~ +1500` (!).
    This is technically fine, because pθ is a probability *density*, but it
    cannot be exp'd without causing overflow, even at float64.

    To overcome this, we use the smoothmax identity for the logarithm of a sum.
    Let `r[s] := pθ(x, z[s]) / qϕ(z[s]|x)`. The identity allows us to express
    `log(∑s r[s])` in terms of the `log(r[s])`::

        log(x + y) = log x + softplus(log y - log x)

    where::

        softplus(x) ≡ log(1 + exp(x))

    To obtain `log(∑s r[s])`, we start from `log r[0]`, and then (using the
    associative property of addition) accumulate over 2-tuples in a loop
    (which we vectorize for speed).

    The final detail is to handle the global scaling factor in the MC
    representation of the expectation, but this is easy::

       log(α x) = log α + log x

    so that we actually evaluate::

        log(α ∑s r[s]) = log α + log(∑s r[s])

    The definition of the NLL metric is given e.g. in Sinha and Dieng (2022):
      https://arxiv.org/pdf/2105.14859.pdf
    The smoothmax identity for the logarithm of a sum is discussed e.g. in:
      https://cdsmithus.medium.com/the-logarithm-of-a-sum-69dd76199790
    """
    print("NLL: encoding...")
    mean, logvar = model.encoder.predict(x, batch_size=batch_size)

    @batched(batch_size)
    def samplewise_elbo(x, mean, logvar):  # positional parameters get @batched
        """log(pθ(x, z) / qϕ(z|x)) for each sample of x, drawing one MC sample of z.

        `x`: tensor of shape (N, 28, 28, 1); data batch of grayscale pictures
        `mean`, `logvar`: output of encoder with input `x`

        Returns a tensor of shape (N,).
        """
        ignored_eps, z = model.reparameterize(mean, logvar)  # draw MC sample
        # Rewriting the joint probability:
        #   pθ(x, z) = pθ(x|z) pθ(z)
        # we have
        #   log pθ(x, z) = log(pθ(x|z) pθ(z))
        #                = log pθ(x|z) + log pθ(z)
        logpx_z = log_px_z(model, x, z, training=False)  # log pθ(x|z)
        logpz = log_normal_pdf(z, 0., 0.)                # log pθ(z)
        logpxz = logpx_z + logpz                         # log pθ(x, z)

        logqz_x = log_normal_pdf(z, mean, logvar)        # log qϕ(z|x)

        return logpxz - logqz_x

    print(f"NLL: MC sampling (n = {n_mc_samples})...")
    acc = [samplewise_elbo(x, mean, logvar) for _ in range(n_mc_samples)]  # -> [[N], [N], ...]
    acc = tf.stack(acc, axis=1)  # -> [N, n_mc_samples]

    # TODO: I think this is correct, we should reduce linearly here. But check just to be sure.
    # Taking the mean like this computes (albeit averaging over `x` too early; strictly, we should accumulate the MC samples first):
    #   -E[x ~ data](log E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ))
    # whereas treating both dimensions the same (flatten, send to accumulation loop) would compute:
    #   -log E[x ~ data](E[z ~ qϕ(z|x)]( pθ(x, z) / qϕ(z|x) ))
    acc = tf.reduce_mean(acc, axis=0)  # -> [n_mc_samples]

    print("NLL: computing MC estimate...")
    # `log(∑k r[k])`, in terms of `log r[k]`, using the smoothmax identity.
    out = logsum(acc)
    out += tf.math.log(1. / float(tf.shape(acc)))  # scaling in the expectation operator
    return -out.numpy()  # *negative* log-likelihood


def mutual_information(model, x, *, batch_size=1024, n_mc_samples=10):
    """[performance statistic] Compute mutual information between x and its code z.

    We actually compute and return two related metrics; the KL regularization
    term of the ELBO, namely the KL divergence of the variational posterior
    from the latent prior::

        E[x ~ pd(x)]( DKL[qϕ(z|x) ‖ pθ(z)] )

    and the mutual information induced by the variational joint, defined as::

        I[q](x, z) := E[x ~ pd(x)]( DKL[qϕ(z|x) ‖ pθ(z)] - DKL[qϕ(z) ‖ pθ(z)] )

    where DKL is the Kullback-Leibler divergence::

        DKL[q(z) ‖ p(z)] ≡ E[z ~ q(z)]( log q(z) - log p(z) ),

    pd(x) is the empirical data distribution, and qϕ(z) is the aggregated posterior,
    which is the marginal over z induced by the joint::

        qϕ(z, x) := qϕ(z|x) pd(x)

    or in other words,

        qφ(z) ≡ ∫ qφ(z|x) pd(x) dx

    The return value is `(DKL, MI)`, approximated using Monte Carlo.

    Slightly different definitions the MI metric are given e.g. in
    Sinha and Dieng (2022), and in Dieng et al. (2019):
      https://arxiv.org/pdf/2105.14859.pdf
      https://arxiv.org/pdf/1807.04863.pdf
    """
    # TODO: Refactor this, useful as-is (the "KL" metric reported in some papers).
    #
    # TODO: Investigate: if qϕ(z|x) and pθ(z) are both gaussian (as they are here), Takahashi et al. (2019)
    #       note that it should be possible to evaluate the KL divergence in closed form, citing the
    #       original VAE paper by Kingma and Welling (2013):
    #       [Kingma and Welling 2013] Kingma, D. P., and Welling, M.2013. Auto-encoding variational Bayes.
    #       arXiv preprint arXiv:1312.6114.
    def dkl_qz_x_from_pz(x, n_mc_samples):
        """Estimate the KL divergence of the variational posterior from the latent prior.

        This is the KL regularization term that appears in the ELBO
        (in one of its alternative expressions).

        Defined as::
            DKL(qϕ(z|x) ‖ pθ(z)) ≡ E[z ~ qϕ(z|x)]( log qϕ(z|x) - log pθ(z) )

        `n_z_mc_samples` is the number of Monte Carlo samples to use to
        estimate the expectation.
        """
        mean, logvar = model.encoder.predict(x, batch_size=batch_size)  # prevent re-batching into smaller subbatches
        @batched(batch_size)
        def logratio(x, mean, logvar):  # positional parameters get @batched
            # Given an `x`, we draw a single MC sample of `z ~ qϕ(z|x)`, where the distribution is given by the encoder.
            ignored_eps, z = model.reparameterize(mean, logvar)
            logqz_x = log_normal_pdf(z, mean, logvar)  # log qϕ(z|x)
            logpz = log_normal_pdf(z, 0., 0.)          # log pθ(z)
            return logqz_x - logpz

        # E[z ~ qϕ(z|x)](...)
        out = [logratio(x, mean, logvar) for _ in range(n_mc_samples)]  # [N, n_mc_samples]
        out = tf.reduce_mean(out, axis=1)  # [N]
        return out

    # The first DKL term. For each given `x`:
    #   DKL(qϕ(z|x) ‖ pθ(z)) ≡ E[z ~ qϕ(z|x)]( log qϕ(z|x) - log pθ(z) )
    first_dkl_term = dkl_qz_x_from_pz(x, n_mc_samples)  # [N]

    # The second DKL term (note this does not depend on `x`):
    #   DKL(qϕ(z) ‖ pθ(z)) = E[z ~ qϕ(z)]( log qϕ(z) - log pθ(z) ),
    #
    # For this, we need access to the aggregated posterior qϕ(z). Ouch!
    #
    # If a joint distribution is available, it is possible to obtain samples from a marginal by sampling the joint,
    # by just ignoring the other outputs:
    #   https://math.stackexchange.com/questions/3236982/marginalizing-by-sampling-from-the-joint-distribution
    #
    # But even better here is:
    #
    # "We can sample from p(z) and qφ(z|x) since these distributions are a Gaussian, and we can also sample from
    # the aggregated posterior qφ(z) by using ancestral sampling: we choose a data point x from a dataset randomly
    # and sample z from the encoder given this data point x."
    #   --Takahashi et al. (2019):
    #     https://arxiv.org/pdf/1809.05284.pdf
    #
    # That gives us a single Monte Carlo sample of `z ~ qφ(z)`, which we can plug into `log qφ(z)` (and thus into
    # the second DKL term). To get the expectation, we average this MC sampling over `z` the usual way.
    #
    # For each sample `z`, how to actually evaluate the log-density `log qφ(z)`? The aggregated posterior is:
    #
    #   qφ(z) ≡ ∫ qφ(z|x) pd(x) dx      (marginalize away `x` in the joint `qϕ(z, x) = qϕ(z|x) pd(x)`)
    #         = E[x ~ pd(x)](qφ(z|x))
    #         ≈ (1/K) ∑k qφ(z|xk)       (Monte Carlo estimate)
    #
    # where, since `x ~ pd(x)`, we just draw K samples `xk` from the dataset. Taking the `log`,
    #
    #   log qφ(z) ≈ log( (1/K) ∑k qφ(z|xk) )
    #             = log(1/K) + log(∑k qφ(z|xk))
    #
    # We can use the smoothmax identity to evaluate the logarithm of the sum in terms of `log qφ(z|xk)`.
    # We evaluate each `log qφ(z|xk)` at the already sampled values `z ~ qφ(z)` and `xk ~ pd(x)`.
    # In practice:
    #    - Use the already sampled `z`
    #    - Use (μ, log σ) of the model corresponding to each sampled `xk`
    #
    # TODO: refactor; there's a lot of useful stuff here (e.g. `compute_logqz` is useful on its own, to plot the aggregated posterior log-density).
    # TODO: when using the whole dataset for MC samples, we could precompute `mean` and `logvar` in one go for *all* `x`.
    def dkl_qz_from_pz(n_mc_samples):
        """Estimate the KL divergence of the aggregated posterior from the latent prior.

        This term appears in the mutual information (MI).

        Defined as::
            DKL(qϕ(z) ‖ pθ(z)) ≡ E[z ~ qϕ(z)]( log qϕ(z) - log pθ(z) )

        `n_z_mc_samples` is the number of Monte Carlo samples to use to
        estimate the expectation.
        """
        def sample_x(n):
            """Sample `x ~ pd(x)`, i.e. draw random samples from the dataset.

            `n`: how many `x` to return (as tensor of shape (n,))
            """
            # https://stackoverflow.com/questions/50673363/in-tensorflow-randomly-subsample-k-entries-from-a-tensor-along-0-axis
            ks = tf.range(tf.shape(x)[0])
            random_ks = tf.random.shuffle(ks)[:n]
            return tf.gather(x, random_ks)

        def sample_qz(n):
            """Ancestrally sample `z ~ qφ(z)`.

            `n`: how many `z` to return (as tensor of shape (n,))
            """
            # Step 1: randomly pick `n` samples from the dataset.
            xs = sample_x(n)
            # Step 2: sample one `z` for each `x`.
            mean, logvar = model.encoder.predict(xs, batch_size=batch_size)
            zs = model.reparameterize(mean, logvar)
            return zs

        # TODO: vectorize for many `z` at once
        def compute_logqz(z, n_mc_samples):
            """Evaluate aggregated posterior qφ(z) at `z`, using MC sampling."""
            xk = sample_x(n_mc_samples)  # inner MC sample: for evaluation of qφ(z)

            # For each sample `xk`, evaluate `log qϕ(z|xk)`.
            # For this we need the variational posterior parameters, so encode `xk`:
            mean, logvar = model.encoder.predict(xk, batch_size=batch_size)  # each of mean, logvar: [n_mc_samples, latent_dim]

            # z_broadcast = tf.expand_dims(z, axis=0)
            logqz_x = log_normal_pdf(z, mean, logvar)  # log qϕ(z|x)  # [n_mc_samples]

            # `log(∑k qφ(z|xk))`, in terms of `log qφ(z|xk)`, using the smoothmax identity.
            logqz = logsum(logqz_x)
            logqz += tf.math.log(1. / float(tf.shape(logqz_x)))  # scaling in the expectation operator
            return logqz

        # Compute the DKL, and average over the MC samples `(z, log qφ(z))`.
        # With the above definitions, this is as simple as:
        dkls = []
        for z in sample_qz(n_mc_samples):  # outer MC sample: for ancestral sampling of `z`
            logqz = compute_logqz(z, n_mc_samples)
            logpz = log_normal_pdf(z, 0., 0.)
            dkls.append(logqz - logpz)
        dkl = tf.reduce_mean(dkls).numpy()  # evaluate the MC expectation
        return dkl

    second_dkl_term = dkl_qz_from_pz(n_mc_samples)  # just a scalar

    MI = first_dkl_term - second_dkl_term

    # Finally, average over the dataset `x`.
    dkl = tf.reduce_mean(first_dkl_term).numpy()
    MI = tf.reduce_mean(MI).numpy()
    return dkl, MI
