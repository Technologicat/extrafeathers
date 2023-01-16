"""Residual network (ResNet) blocks, implemented as custom Keras layers.

These can be used e.g. for building slightly more advanced NN architectures for autoencoders.

We skip batch normalization for now, because I'm new to the AI field, and not sure (as of
this writing) how to invert a normalizer, which we would need to do to build a symmetric
decoder.

Also, BN won't do anything useful anyway until we set the `training` flag correctly, which
`main.py` currently does not bother with. The network always thinks `training=False`,
which would make BN produce nonsense, since it hasn't been calibrated.

The implementation is based on combining information from:

    https://www.tensorflow.org/tutorials/customization/custom_layers
    https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
    https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557

but mostly this is just a minimalistic jazz solo on the general idea of ResNets, for exploring them
in the context of the MNIST datasets and `extrafeathers`.
"""

__all__ = ["IdentityBlock", "ConvolutionBlock",
           "IdentityBlockTranspose", "ConvolutionBlockTranspose"]

import tensorflow as tf

# --------------------------------------------------------------------------------

# TODO: rename -> IdentityBlock2D
class IdentityBlock(tf.keras.layers.Layer):
    """A simple ResNet identity block.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input must have `filters` channels so that the skip connection works.
    To work with input with a different number of channels, use e.g. a convolution
    layer with kernel size 1 as an adaptor::

        # adaptor
        x = Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation="relu")(x)

        # now `x` can be fed into a 32-filter `IdentityBlock`
        x = IdentityBlock(filters=32, kernel_size=3)(x)
    """

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        # The purpose of the size-1 convolution is to cheaply change the dimensionality (number of channels)
        # in the filter space, without introducing spatial dependencies:
        #   https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        #
        # In the blocks defined in this module, the activation of the last sublayer is handled in `call`,
        # because we need to add the residual from the skip-connection before applying the activation.
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            padding="same", activation="relu")
        # TODO: conv3 seems useless if we use the same number of filters, it's just remapping channels?
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

# TODO: rename -> ConvolutionBlock2D
class ConvolutionBlock(tf.keras.layers.Layer):
    """A simple ResNet convolution block.

    Tensor sizes::

        [batch, 2*n, 2*n, channels] -> [batch, n, n, filters]

    The convention with `filters` is the same as in `Conv2D`; it's the number
    of *output* channels.
    """

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2,
                                            padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")  # activation handled in `call`
        # Classically, downsampling is done here by a size-1 convolution ignoring 3/4 of the pixels:
        # self.downsample = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2)
        # But perhaps we could try something like this:
        self.downsample = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=2,
                                                                                padding="same"),
                                               tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                                      padding="same")])
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x_skip = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

# --------------------------------------------------------------------------------

# TODO: rename -> IdentityBlockTranspose2D
class IdentityBlockTranspose(tf.keras.layers.Layer):
    """The architectural inverse of `IdentityBlock`, for autoencoder decoders.

    Tensor sizes::

        [batch, n, n, filters] -> [batch, n, n, filters]

    The input must have `filters` channels so that the skip connection works.
    """

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x

# TODO: rename -> ConvolutionBlockTranspose2D
class ConvolutionBlockTranspose(tf.keras.layers.Layer):
    """The architectural inverse of `ConvolutionBlock`, for autoencoder decoders.

    Tensor sizes::

        [batch, n, n, channels] -> [batch, 2*n, 2*n, filters]

    The convention with `filters` is the same as in `Conv2DTranspose`; it's the number
    of *output* channels.
    """

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")  # activation handled in `call`
        self.upsample = tf.keras.Sequential([tf.keras.layers.Upsampling2D(size=2,
                                                                          interpolation="bilinear"),
                                             tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                                                    padding="same")])
        self.adder = tf.keras.layers.Add()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x_skip = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adder([x, x_skip])
        x = self.activation(x)
        return x
