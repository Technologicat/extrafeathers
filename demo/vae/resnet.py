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
           "ConvolutionBlock2D", "ConvolutionBlockTranspose2D"]

import tensorflow as tf

# --------------------------------------------------------------------------------

class ResidualBlock2D(tf.keras.layers.Layer):
    """Classic basic ResNet residual block, with a simple passthrough skip-connection.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input must have `filters` channels so that the skip-connection works.
    """

    def __init__(self, filters, kernel_size, *, name=None, activation=None):
        super().__init__(name=name)
        # In the blocks defined in this module, the activation of the last sublayer is handled in `call`,
        # because we need to add the residual from the skip-connection before applying the activation.
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

class ResidualBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `ResidualBlock2D`, for autoencoder decoders."""

    def __init__(self, filters, kernel_size, *, name=None, activation=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                     padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
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
        # The purpose of the size-1 convolution is to cheaply change the dimensionality (number of channels)
        # in the filter space, without introducing spatial dependencies:
        #   https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        # It also acts as a "feature selector" (since the weights are trainable), and applies a ReLU.
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
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
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
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
        # The purpose of the size-1 convolution is to cheaply change the dimensionality (number of channels)
        # in the filter space, without introducing spatial dependencies:
        #   https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        # It also acts as a "feature selector" (since the weights are trainable), and applies a ReLU.
        #
        # In the blocks defined in this module, the activation of the last sublayer is handled in `call`,
        # because we need to add the residual from the skip-connection before applying the activation.
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")  # activation handled in `call`
        self.projection = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                 padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = self.projection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
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
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")  # activation handled in `call`
        self.projection = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                          padding="same")  # activation handled in `call`
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = self.projection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

# --------------------------------------------------------------------------------

class ConvolutionBlock2D(tf.keras.layers.Layer):
    """A ResNet bottleneck convolution block.

    `strides` is passed to the convolution, and on the skip-connection, the input
    is downsampled using the same `strides` (currently by local average pooling,
    followed by a projection).

    Tensor sizes::

        [batch, n, n, channels] -> [batch, n // strides, n // strides, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.
    """

    def __init__(self, filters, kernel_size, *, strides=2, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=kernel_size,
                                            strides=strides,
                                            padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")  # activation handled in `call`
        # Classically, downsampling is done here by a size-1 convolution ignoring 3/4 of the pixels:
        # self.downsample = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2)
        # But perhaps we could try something like this:
        self.downsample = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=strides,
                                                                                padding="same"),
                                               tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                                      padding="same")])
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

class ConvolutionBlockTranspose2D(tf.keras.layers.Layer):
    """The architectural inverse of `ConvolutionBlock2D`, for autoencoder decoders.

    `strides` is passed to the convolution transpose, and on the skip-connection,
    the input is upsampled using the same `strides` (currently by bilinear interpolation,
    followed by a projection).

    Tensor sizes::

        [batch, n, n, channels] -> [batch, strides*n, strides*n, filters]

    The input passes through a bottleneck of `max(1, filters // bottleneck_factor)`
    channels; the final output has `filters` channels.
    """

    def __init__(self, filters, kernel_size, *, strides=2, name=None, activation=None, bottleneck_factor=4):
        super().__init__(name=name)
        bottleneck = max(1, filters // bottleneck_factor)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=bottleneck, kernel_size=kernel_size,
                                                     strides=strides,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")  # activation handled in `call`
        self.upsample = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=strides,
                                                                          interpolation="bilinear"),
                                             tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                                    padding="same")])
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.activations.get(activation)

    def call(self, x, training=False):
        x_skip = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x
