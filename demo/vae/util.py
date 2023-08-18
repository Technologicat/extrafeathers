"""Various small utilities for the VAE example."""

__all__ = ["layer_to_model",
           "preprocess_images",
           "compute_squared_l2_error",
           "sorted_by_l2_error",
           "batched",
           "delete_directory_recursively",
           "create_directory",
           "clear_and_create_directory"]

from functools import wraps
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
    outputs = layer(inputs)
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

@tf.function
def compute_squared_l2_error(x: tf.Tensor, xhat: tf.Tensor):
    """Compute (‖xhat - x‖_l2)**2. Convenience function.

    `x`, `xhat`: tensor of shape `[N, ny, nx, c]`

    Returns a tensor of shape `[N]`, the squared l2 error for each sample in `x`.
    """
    return tf.reduce_sum((x - xhat)**2, axis=[1, 2, 3])

def sorted_by_l2_error(x: tf.Tensor, xhat: tf.Tensor, *,
                       reverse: bool = False):
    """Given inputs `x` and corresponding predictions `xhat`, sort by l2 error.

    `x`, `xhat`: tensor of shape [N, ny, nx, c]
    `reverse`: If `True`, sort in descending order.
               If `False` (default), sort in ascending order.

    Returns `(e2s, ks)`, where:
      `e2s`: rank-1 `np.array`, length `N`; (‖xhat - x‖_l2)**2 for each sample in `x`.
      `ks`: rank-1 `np.array`, length `N`; indices that sort `e2s`, as in `np.argsort`.
    """
    e2s = compute_squared_l2_error(x, xhat).numpy()
    objective = -e2s if reverse else e2s
    return e2s, np.argsort(objective, kind="stable")

# --------------------------------------------------------------------------------

# TODO: Surely there is something like this already available in TensorFlow, but I can't find it?
# (`Model.fit` et al. have a `batch_size` parameter, which goes to a dataset handler, which also takes a bunch of unrelated parameters,
#  but there doesn't seem to be a batched evaluation wrapper for math operations.)
def batched(batch_size: int) -> typing.Callable:
    """Parametric decorator: transparently batch some TensorFlow math.

    Like the `batch_size` argument in `Model.fit` et al., but for arbitrary
    TF math on tensor(s) where the first axis indexes the dataset.

    This is useful when the operation on a full dataset won't fit in VRAM,
    and needs to be computed in batches.

    The operation may take any number of tensors as positional arguments,
    and any keyword arguments. Only the positional arguments are batched.

    The operation may return either a single tensor, or a tuple of tensors.
    In the output, the batches are reassembled.

    Usage::

        @batched(1024)
        def op(x, y, *, furble):
            # ... do tf math on tensors here ...
            return T  # or return T0, T1, ...
        op(x, y, furble=True)  # now this computes in batches of 1024
    """
    def batcher(f: typing.Callable) -> typing.Callable:  # batcher, batcher, mushroom
        @wraps(f)
        def evaluate_batched(*args, **kwargs):
            batched_args = [tf.data.Dataset.from_tensor_slices(tensor).batch(batch_size) for tensor in args]  # [T0, T1, ...] -> [batches(c0), batches(c1), ...]
            batches_in = [[batch for batch in batches] for batches in batched_args]  # -> [[b0_T0, b1_T0, ...], [b0_T1, b1_T1, ...], ...]
            batches_in = list(zip(*batches_in))  # -> [[b0_T0, b0_T1, ...], [b1_T0, b1_T1, ...], ...]
            batches_out = [f(*xs, **kwargs) for xs in batches_in]
            has_multiple_outputs = batches_out and isinstance(batches_out[0], tuple)
            if not has_multiple_outputs:  # single tensor
                return tf.concat(batches_out, axis=0)  # [b0_T, b1_T, ...] -> T
            components_out = list(zip(*batches_out))  # [[b0_T0, b0_T1, ...], [b1_T0, b1_T1, ...], ...] -> [[b0_T0, b1_T0, ...], [b0_T1, b1_T1, ...], ...]
            return tuple(tf.concat(x, axis=0) for x in components_out)  # -> [T0, T1, ...]
        return evaluate_batched
    return batcher

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
