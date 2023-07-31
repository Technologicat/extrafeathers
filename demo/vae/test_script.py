#!/usr/bin/python
#
# Notes for interactive testing in an IPython session.
#
# For the main scripts, see `main.py` (model training)
# and `anim.py` (create animations from completed training).

import gc
import importlib

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

# Load a model snapshot:
main.model = tf.keras.models.load_model(f"{config.output_dir}model/final.keras")  # or whatever
# main.model.my_load(f"{config.output_dir}model/final")  # to load a snapshot produced by the legacy custom saver

# plt.ion()  # interactive mode doesn't seem to work well with our heavily customized overlay plot

e = plotter.plot_latent_image(21)
plotter.overlay_datapoints(train_images, train_labels, e)

fig = plt.figure(1)
fig.savefig(f"{config.output_dir}{config.overlay_fig_basename}_final_from_test_script.{config.fig_format}")
fig.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

plt.show()
