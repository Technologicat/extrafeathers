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
import demo.vae.plotter as plotter
import demo.vae.util as util

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = util.preprocess_images(train_images)
test_images = util.preprocess_images(test_images)

# Reset everything, so that we can run this snippet multiple times in the same IPython session
# (useful for exploring different snapshots of the model).
plt.close(1)
importlib.reload(main)  # re-instantiate CVAE
tf.keras.backend.clear_session()  # clean up dangling tensors
gc.collect()  # and make sure they are gone

main.model.my_load("demo/output/vae/model/final")  # or whatever

# plt.ion()  # interactive mode doesn't seem to work well with our heavily customized overlay plot

e = plotter.plot_latent_image(21)
plotter.overlay_datapoints(train_images, train_labels, e)

# fig = plt.figure(1)
# fig.savefig("temp.png")
# fig.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

plt.show()
