"""Various small utilities for the VAE example."""

__all__ = ["layer_to_model",
           "preprocess_images",
           "delete_directory_recursively",
           "create_directory",
           "clear_and_create_directory"]

import os
import pathlib
import typing

import numpy as np

import tensorflow as tf

# --------------------------------------------------------------------------------

def layer_to_model(layer: tf.keras.layers.Layer,
                   input_shape: typing.Tuple[typing.Optional[int]]) -> tf.keras.Model:
    """Convert a `Layer` to a `Model` to make it inspectable.

    This allows visualizing the internal structure of a custom layer with `.summary()`
    or `tf.keras.utils.plot_model` when that layer is modular enough (e.g. its `call`
    method consists of calls to the Keras functional API).

    For example::

        from demo.vae.resnet import ConvolutionBlock2D

        layer = ConvolutionBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                                   bottleneck_factor=2)
        model = layer_to_model(layer, input_shape=(28, 28, 1))
        model.summary()
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")
    outputs = layer.call(inputs)
    return tf.keras.Model(inputs, outputs)

# --------------------------------------------------------------------------------

def preprocess_images(images, digit_size=28, channels=1, discrete=False):
    """Preprocess square images, 8 bits per channel, for use with the CVAE.

    Reshape to `(?, digit_size, digit_size, channels)`.

    Scale each channel to [0, 1], by dividing by 255.

    If `discrete=True`, binarize to {0, 1} to make compatible with a discrete Bernoulli
    observation model. Otherwise keep as continuous, compatible with a continuous
    Bernoulli observation model.

    Return as a `float32` array.
    """
    images = images.reshape((images.shape[0], digit_size, digit_size, channels)) / 255.
    if discrete:
        images = np.where(images > .5, 1.0, 0.0)
    # # We could also do further modifications here:
    # # https://www.tensorflow.org/tutorials/images/data_augmentation
    # IMG_SIZE = 32
    # resize_and_rescale = tf.keras.Sequential([
    #   layers.Resizing(IMG_SIZE, IMG_SIZE),
    #   layers.Rescaling(1./255)
    # ])
    # images = resize_and_rescale(images)
    return images.astype("float32")

# --------------------------------------------------------------------------------

def delete_directory_recursively(path: str) -> None:
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

def create_directory(path: str) -> None:
    p = pathlib.Path(path).expanduser().resolve()
    pathlib.Path.mkdir(p, parents=True, exist_ok=True)

def clear_and_create_directory(path: str) -> None:
    delete_directory_recursively(path)
    create_directory(path)
