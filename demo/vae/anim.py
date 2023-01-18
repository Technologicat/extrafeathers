#!/usr/bin/python
"""Plot of the evolution of the code points corresponding to the dataset.

Also, make animations.

This script needs a trained VAE model (see `main.py`). After the training finishes,
run this script from the top-level `extrafeathers` directory as:

    python -m demo.vae.anim

!!! Run this script on the CPU; the GPU will easily run out of memory when plotting dataset evolution. !!!
"""

# TODO: add an `ETAEstimator`, like in `main.py`

import gc
import glob
import importlib

import imageio

from unpythonic import ETAEstimator

import tensorflow as tf

import matplotlib.pyplot as plt

from . import main as main
from .config import (output_dir, fig_format,
                     test_sample_fig_basename, test_sample_anim_filename,
                     latent_space_fig_basename, latent_space_anim_filename,
                     overlay_fig_basename, overlay_anim_filename)
from .plotter import plot_latent_image, overlay_datapoints
from .util import preprocess_images

# TODO: refactor dataset config too into `config.py`
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# --------------------------------------------------------------------------------
# Animation-making helper

def make_animation(output_filename: str, input_glob: str) -> None:
    with imageio.get_writer(output_filename, mode="I") as writer:
        filenames = sorted(glob.glob(input_glob))
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)

# --------------------------------------------------------------------------------
# Easy ones first (the training process already writes the individual animation frames):

make_animation(f"{output_dir}{test_sample_anim_filename}",
               f"{output_dir}{test_sample_fig_basename}*.{fig_format}")
make_animation(f"{output_dir}{latent_space_anim_filename}",
               f"{output_dir}{latent_space_fig_basename}*.{fig_format}")

# --------------------------------------------------------------------------------
# Plot the evolution of the codepoints corresponding to the training dataset

input_dirs = sorted(glob.glob(f"{output_dir}model/*"))
est = ETAEstimator(len(input_dirs), keep_last=10)
for model_dir in input_dirs:
    epoch_str = model_dir[model_dir.rfind("/") + 1:]

    # reset the model
    importlib.reload(main)  # re-instantiate CVAE
    tf.keras.backend.clear_session()  # clean up dangling tensors
    gc.collect()  # and make sure they are gone

    # Now we should be able to load a different checkpoint
    main.model.my_load(model_dir)

    plt.close(1)

    try:
        epoch = int(epoch_str)
    except ValueError:
        epoch = None

    e = plot_latent_image(21, epoch=epoch)
    overlay_datapoints(train_images, train_labels, e)

    fig = plt.figure(1)
    fig.savefig(f"{output_dir}{overlay_fig_basename}_{epoch_str}.{fig_format}")
    fig.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

    est.tick()
    print(f"Done {model_dir}, walltime {est.formatted_eta}")

# --------------------------------------------------------------------------------
# Finally, make an animation of the codepoint evolution:

make_animation(f"{output_dir}{overlay_anim_filename}",
               f"{output_dir}{overlay_fig_basename}*.{fig_format}")
