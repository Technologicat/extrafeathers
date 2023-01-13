#!/usr/bin/python
#
# Notes for interactive testing in an IPython session.

import gc
import importlib
import tensorflow as tf
import matplotlib.pyplot as plt
import demo.vae.main as main

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = main.preprocess_images(train_images)
test_images = main.preprocess_images(test_images)

# Reset everything, so that we can run this snippet multiple times in the same IPython session.
plt.close(1)
importlib.reload(main)  # re-instantiate CVAE
tf.keras.backend.clear_session()  # clean up dangling tensors
gc.collect()  # and make sure they are gone

main.model.my_load("demo/output/vae/model/final")  # or whatever

# plt.ion()  # interactive mode doesn't seem to work well with our heavily customized overlay plot

e = main.plot_latent_image(21)
main.overlay_datapoints(train_images, train_labels, e)

# fig = plt.figure(1)
# fig.savefig("temp.png")
# fig.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

plt.show()
