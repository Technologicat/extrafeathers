#!/usr/bin/python
#
# Notes for interactive testing in an IPython session.
#
# For the main scripts, see `main.py` (model training)
# and `anim.py` (create animations from completed training).

import gc
import importlib
import sys

import matplotlib.pyplot as plt

import tensorflow as tf

import demo.vae.main as main
import demo.vae.config as config
import demo.vae.plotter as plotter
import demo.vae.util as util

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = util.preprocess_images(train_images)
test_images = util.preprocess_images(test_images)

# Be aware that as of this writing, (January 2023, tf 2.12-nightly), TensorFlow
# can be finicky with regard to resetting, if you wish to plot different model
# instances during the same IPython session (which is useful for exploring
# different snapshots of the model).
#
# So let's reset everything:
plt.close(1)
importlib.reload(main)  # re-instantiate CVAE
tf.keras.backend.clear_session()  # clean up dangling tensors
gc.collect()  # and make sure they are gone

# Optionally, take a snapshot ID from the command line.
# E.g. "0153" for epoch 153, or "final" for the final after-training state.
if len(sys.argv) > 1:
    snapshot = sys.argv[1]
else:
    snapshot = "final"

# Load a model snapshot:
main.model = tf.keras.models.load_model(f"{config.output_dir}model/{snapshot}.keras")  # or whatever
# main.model.my_load(f"{config.output_dir}model/final")  # to load a snapshot produced by the legacy custom saver

# plt.ion()  # interactive mode doesn't seem to work well with our heavily customized overlay plot

if main.model.latent_dim == 2:
    latent_image = plotter.plot_latent_image(21)
    plotter.overlay_datapoints(train_images, train_labels, latent_image)
else:
    # n = 4000  # in practice fine
    n = test_images.shape[0]  # using all of the data is VERY slow (~10 minutes), but gives the best view.
    plotter.plot_manifold(test_images[:n, :, :, :], test_labels[:n], methods="all")

fig = plt.figure(1)
fig.savefig(f"{config.output_dir}{config.overlay_fig_basename}_{snapshot}_from_test_script.{config.fig_format}")
fig.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

plt.show()
