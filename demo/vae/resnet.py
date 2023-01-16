"""Residual network (ResNet) blocks for building slightly more advanced autoencoder architectures."""

# We skip batch normalization for now because I'm not sure how to invert that in the decoder.
# Based on combining information from:
#   https://www.tensorflow.org/tutorials/customization/custom_layers
#   https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
#   https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557

__all__ = ["IdentityBlock", "ConvolutionBlock",
           "IdentityBlockTranspose", "ConvolutionBlockTranspose"]

import tensorflow as tf

# --------------------------------------------------------------------------------

class IdentityBlock(tf.keras.layers.Layer):
    """[batch, n, n, filters] -> [batch, n, n, filters]

    The input must have `filters` channels so that the skip connection works.
    """

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            padding="same", activation="relu")
        # Activation of the last layer is handled in `call`.
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

class ConvolutionBlock(tf.keras.layers.Layer):
    """[batch, 2*n, 2*n, channels] -> [batch, n, n, filters]"""

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2,
                                            padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                            padding="same")  # activation handled in `call`
        # Classically, downsampling is done here by a convolution skipping 3/4 of the pixels:
        # self.downsample = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2)
        # But perhaps we could try something like this:
        self.downsample = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=2, padding="same"),
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

class IdentityBlockTranspose(tf.keras.layers.Layer):
    """[batch, n, n, filters] -> [batch, n, n, filters]"""

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

class ConvolutionBlockTranspose(tf.keras.layers.Layer):
    """[batch, n, n, filters] -> [batch, 2*n, 2*n, filters]"""

    def __init__(self, filters, kernel_size, *, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,
                                                     padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=1,
                                                     padding="same")  # activation handled in `call`
        self.upsample = tf.keras.layers.Upsampling2D(size=2, interpolation="bilinear")
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
