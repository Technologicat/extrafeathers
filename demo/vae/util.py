"""Various small utilities for the VAE example."""

__all__ = ["preprocess_images",
           "delete_directory_recursively",
           "create_directory",
           "clear_and_create_directory"]

import os
import pathlib

import numpy as np

# --------------------------------------------------------------------------------

def preprocess_images(images, digit_size=28, discrete=False):
    """Preprocess MNIST dataset."""
    images = images.reshape((images.shape[0], digit_size, digit_size, 1)) / 255.  # scale to [0, 1]
    if discrete:  # binarize to {0, 1}, to make compatible with discrete Bernoulli observation model
        images = np.where(images > .5, 1.0, 0.0)
    # else continuous grayscale data
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
