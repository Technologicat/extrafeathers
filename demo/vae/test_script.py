#!/usr/bin/python
#
# Notes for interactive testing in an IPython session.

import importlib
import tensorflow as tf
import matplotlib.pyplot as plt
import demo.vae.main as main

saved_model_dir = "demo/output/vae_test6/my_model"  # or whatever

importlib.reload(main)  # so we can simply run this snippet again to refresh the code (and reset the model)
main.model.my_load(saved_model_dir)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = main.preprocess_images(train_images)
test_images = main.preprocess_images(test_images)

plt.ion()

e = main.plot_latent_image(20)
main.overlay_datapoints(train_images, train_labels, e)
