"""Residual network (ResNet) blocks, implemented as custom Keras layers.

These can be used e.g. for building slightly more advanced NN architectures for autoencoders.

We skip batch normalization for now. BN won't do anything useful until we set the `training`
flag correctly, which `main.py` currently does not bother with. The network always thinks
`training=False`, which would make BN produce nonsense, since it hasn't been calibrated.

The implementation is based on combining information from:

    https://www.tensorflow.org/tutorials/customization/custom_layers
    https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
    https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557
    https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02

but mostly this is just a minimalistic jazz solo on the general idea of ResNets, for exploring them
in the context of the MNIST datasets and `extrafeathers`.
"""

__all__ = ["ResidualBlock2D", "ResidualBlockTranspose2D",
           "IdentityBlock2D", "IdentityBlockTranspose2D",
           "ProjectionBlock2D", "ProjectionBlockTranspose2D",
           "ConvolutionBlock2D", "ConvolutionBlockTranspose2D",
           "GNDropoutRegularization"]

from functools import wraps

import tensorflow as tf

from unpythonic import safeissubclass

# TODO: Downsampling type? Max pooling seems more popular than average pooling. Explore why.
# TODO: Upsampling type? We use bilerp; nearest-neighbor would be a better match if we switch to max-pooling.

# TODO: Add batch normalization (BN) to the resnet blocks to allow building deeper nets.
# But beware, BN can misfire in non-obvious ways:
#   https://mindee.com/blog/batch-normalization/
#   https://www.alexirpan.com/2017/04/26/perils-batch-norm.html

# --------------------------------------------------------------------------------

# HACK: the basic activation functions returned by `tf.keras.activations.get` do not take
# the `training` kwarg, but the activation layers in `tf.keras.layers` do. Ideally, we should
# inspect the call signature, and pass on compatible args only, but discarding everything
# except the input tensor `x` will do for now.
#
# Our `call` methods always pass the `training` kwarg, so they work correctly with
# activation layers. We only use this hack as an adaptor for the basic activation functions.
def pass_first_arg_only_to(f):
    """Decorator. Return a copy of `f` that ignores all but the first argument."""
    @wraps(f)
    def adaptor(x, *args, **kwargs):
        return f(x)
    return adaptor

# --------------------------------------------------------------------------------

class ResidualBlock2D(tf.keras.layers.Layer):
    """Classic basic ResNet residual block, with a simple passthrough skip-connection.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input must have `filters` channels so that the skip-connection works.

    `activation`: One of: a function that creates an activation layer when called
                  with no arguments (such as a constructor; e.g. `tf.keras.layers.PReLu`),
                  or a string (see `keras.activations`).

                  If you need to pass parameters to an activation layer,
                  use partial application::

                      from functools import partial

                      my_activation = partial(tf.keras.layers.PReLU,
                                              shared_axes=...)

                  and then pass `my_activation` as the `activation` parameter here.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None):
        super().__init__(name=name)

        # In the blocks defined in this module:
        #
        # The activation of the last sublayer is handled after adding the residual
        # from the skip-connection.
        #
        # We use PReLU (trainable leaky ReLU) activation instead of the basic ReLU,
        # as it tends to perform slightly better, at almost no extra computational cost.
        # See He et al. (2015):
        #   https://arxiv.org/abs/1502.01852
        #
        # We use "he_normal" (a.k.a. Kaiming) initialization of kernel weights instead of the default
        # "glorot_uniform" (a.k.a. Xavier) initialization, because He performs better for initialization
        # of convolution kernels in deep networks.
        #
        # See e.g.:
        #   https://keras.io/api/layers/convolution_layers/convolution2d/
        #   https://keras.io/api/layers/initializers/#henormal-class
        #   https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are
        #   https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization

        # main path
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            kernel_initializer="he_normal",
                                            padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act2 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.adder([x, inputs], *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        return x

class ResidualBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `ResidualBlock2D`, for autoencoder decoders."""

    def __init__(self, filters, kernel_size, *, name=None, activation=None):
        super().__init__(name=name)

        # main path
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                     kernel_initializer="he_normal",
                                                     padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act2 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.adder([x, inputs], *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        return x

# --------------------------------------------------------------------------------

class IdentityBlock2D(tf.keras.layers.Layer):
    """A ResNet bottleneck identity block, with a simple passthrough skip-connection.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.

    The input must have `filters` channels so that the skip-connection works.
    When this holds, this block is cheaper to use than `ProjectionBlock2D`.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        #
        # The purpose of the size-1 convolution is to cheaply change the dimensionality (number of channels)
        # in the filter space, without introducing spatial dependencies:
        #   https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        #
        # The first size-1 convolution here:
        #   - Acts as a "feature selector" (since the weights are trainable).
        #   - Lowers the computational cost of the block, because it is cheaper to select channels first,
        #     and then perform the size-n convolution on the result, than it is to perform the size-n
        #     convolution directly on the input.
        #
        # The final size-1 convolution then postprocesses the result to output the desired number of features.
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x = self.adder([x, inputs], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

class IdentityBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `IdentityBlock2D`, for autoencoder decoders.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.

    The input must have `filters` channels so that the skip-connection works.
    When this holds, this block is cheaper to use than `ProjectionBlockTranspose2D`.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x = self.adder([x, inputs], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

# --------------------------------------------------------------------------------

class ProjectionBlock2D(tf.keras.layers.Layer):
    """A ResNet bottleneck identity block, with a projection on the skip-connection.

    The projection allows the block to work with any number of input channels.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, n, n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")

        # skip-connection
        self.projection = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                 kernel_initializer="he_normal",
                                                 padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x_skip = self.projection(inputs, *args, **kwargs)
        x = self.adder([x, x_skip], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

class ProjectionBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `ProjectionBlock2D`, for autoencoder decoders.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, n, n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")

        # skip-connection
        self.projection = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                          kernel_initializer="he_normal",
                                                          padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x_skip = self.projection(inputs, *args, **kwargs)
        x = self.adder([x, x_skip], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

# --------------------------------------------------------------------------------

class ConvolutionBlock2D(tf.keras.layers.Layer):
    """A ResNet bottleneck convolution block.

    `strides` is passed to the convolution, and on the skip-connection, the input
    is downsampled using the same `strides` (currently by local average pooling,
    followed by a projection to the desired number of filters).

    `dilation_rate` is also passed to the convolution, so as an alternative mode,
    this supports a dilated (a.k.a. atrous) convolution. This expands the field
    of view (a.k.a. receptive field) of the convolution without increasing the
    amount of computation, at the cost of introducing blind spots (holes, /trous/)
    in the kernel. For example, `dilation_rate=2` computes in a 5×5 region, while
    only sampling 3×3 pixels.

    When `dilation_rate != 1`, `strides` must be `1`.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, n // strides, n // strides, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.

    On the atrous convolution, see Chen et al. (2017), section 3.1:
        https://arxiv.org/pdf/1606.00915v2.pdf
    """

    # TODO: Change the default strides to 1, to match `tf.keras.layers.Conv2D`.
    def __init__(self, filters, kernel_size, *, strides=2, dilation_rate=1, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            kernel_initializer="he_normal",
                                            strides=strides, dilation_rate=dilation_rate,
                                            padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            kernel_initializer="he_normal",
                                            padding="same")

        # skip-connection
        #
        # Classically, downsampling is done here by a size-1 convolution ignoring 3/4 of the pixels:
        # self.downsample = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2)
        # But perhaps we could try something like this:
        # self.downsample = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=strides,
        #                                                                         padding="same"),
        #                                        tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
        #                                                               kernel_initializer="he_normal",
        #                                                               padding="same")])
        # Using `Sequential` hides the details from `summary` (see `util.layer_to_model`).
        # To expose the architecture for inspection, we're better off defining this with individual layers:
        self.downsample = tf.keras.layers.AveragePooling2D(pool_size=strides, padding="same")
        self.compat = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                             kernel_initializer="he_normal",
                                             padding="same")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x_skip = self.downsample(inputs, *args, **kwargs)
        x_skip = self.compat(x_skip, *args, **kwargs)
        x = self.adder([x, x_skip], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

class ConvolutionBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `ConvolutionBlock2D`, for autoencoder decoders.

    `strides` is passed to the convolution transpose, and on the skip-connection,
    the input is upsampled using the same `strides` (currently by projection to
    the desired number of filters, then followed by a bilinear interpolation).

    `dilation_rate` is also passed to the convolution transpose, so as an alternative
    mode, this supports a dilated (a.k.a. atrous) convolution. This expands the field
    of view (a.k.a. receptive field) of the convolution without increasing the
    amount of computation, at the cost of introducing blind spots (holes, /trous/)
    in the kernel. For example, `dilation_rate=2` computes in a 5×5 region, while
    only sampling 3×3 pixels.

    When `dilation_rate != 1`, `strides` must be `1`.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, strides*n, strides*n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.

    On the atrous convolution, see Chen et al. (2017), section 3.1:
        https://arxiv.org/pdf/1606.00915v2.pdf
    """

    def __init__(self, filters, kernel_size, *, strides=2, dilation_rate=1, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)

        # main path
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")
        self.act1 = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     kernel_initializer="he_normal",
                                                     strides=strides, dilation_rate=dilation_rate,
                                                     padding="same")
        self.act2 = tf.keras.layers.PReLU()
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     kernel_initializer="he_normal",
                                                     padding="same")

        # skip-connection
        self.compat = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                      kernel_initializer="he_normal",
                                                      padding="same")
        self.upsample = tf.keras.layers.UpSampling2D(size=strides, interpolation="bilinear")

        # output
        self.adder = tf.keras.layers.Add()
        self.act3 = activation() if safeissubclass(activation, tf.keras.layers.Layer) else pass_first_arg_only_to(tf.keras.activations.get(activation))

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs, *args, **kwargs)
        x = self.act1(x, *args, **kwargs)
        x = self.conv2(x, *args, **kwargs)
        x = self.act2(x, *args, **kwargs)
        x = self.conv3(x, *args, **kwargs)
        x_skip = self.compat(inputs, *args, **kwargs)
        x_skip = self.upsample(x_skip, *args, **kwargs)
        x = self.adder([x, x_skip], *args, **kwargs)
        x = self.act3(x, *args, **kwargs)
        return x

# --------------------------------------------------------------------------------

class GNDropoutRegularization(tf.keras.layers.Layer):
    """Normalization and regularization block.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, n, n, channels]

    Based on group normalization (GN) and spatial dropout (a.k.a. channel dropout),
    which drops entire feature maps (channels).

    This is a convenience class that just encapsulates these two operations into
    a single logical layer in the layout diagram.

    `groups`: Number of groups for GN. If you want instance normalization (IN),
              set this to the number of input `channels`.
    `rate`:   The probability at which each channel is dropped in spatial dropout.
              The default `0.1` is a rule-of-thumb recommendation from Cai et al. (2020).

    See:

        Cai et al. (2020), figure 3: operation ordering: better results if GN first,
        then dropout, just before feeding into the next convolution. The dropout rate
        recommendation (channel retain ratio 0.9) is near the end of section 3.2.3.
            https://arxiv.org/pdf/1904.03392.pdf

        Wu and He (2018), figure 2: instance normalization: normalize over whole image,
        independently in each channel. In Keras, use the GN layer and configure appropriately.
            https://arxiv.org/pdf/1803.08494.pdf
            https://keras.io/api/layers/normalization_layers/group_normalization/

        Tompson et al. (2015), section 3.2: spatial dropout (a.k.a. drop-channel) drops
        whole feature maps, which is useful when nearby pixels are correlated.
            https://arxiv.org/abs/1411.4280
    """

    def __init__(self, groups, rate=0.1, *, name=None):
        super().__init__(name=name)
        self.groupnorm = tf.keras.layers.GroupNormalization(groups=groups)
        self.dropout = tf.keras.layers.SpatialDropout2D(rate=rate)

    def call(self, inputs, *args, **kwargs):
        x = self.groupnorm(inputs, *args, **kwargs)
        x = self.dropout(x, *args, **kwargs)
        return x
