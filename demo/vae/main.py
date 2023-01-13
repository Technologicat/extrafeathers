#!/usr/bin/python
"""Convolutional variational autoencoder (CVAE) implemented in Keras and TensorFlow.

This is a simple technology demo, condensed and improved from existing online VAE tutorials,
that just approximates MNIST digits. The goal of introducing a VAE in the context of
`extrafeathers` is to eventually make a PINN (physically informed neural network) for
the acceleration of numerical simulations in digital twin applications.

The idea is that a PDE solution field, projected onto a uniform grid, can be thought of as
a multichannel image (like an RGB image, but tensor components instead of colors). A PINN
is produced by adding an extra term to the loss function, penalizing the residual of the PDE,
as evaluated using the reconstructed output. The encoder part of the VAE then acts as a
lossy data compressor, reducing the solution field into a low-dimensional representation
(which can be decompressed using the decoder part of the VAE).

For the acceleration of simulations, a VAE is a particularly useful type of neural network,
because its latent representation is continuous. Given a trained VAE, we can train another
neural network to act as a time evolution operator in the low-dimensional latent space, thus
yielding a ROM (reduced order model) for real-time simulation.

Note that a VAE is essentially an interpolator, so it only helps with deployed models, not in
research into new situations. For the output to be meaningful, the situation being simulated,
or a reasonable approximation thereof, must exist in the training data of the VAE. This
limitation is not too different from classical ROM techniques such as proper orthogonal
decomposition (POD); that too is an interpolator, which can only reproduce behavior that
exists in the input data.

As development proceeds, the next planned step is to implement a CVAE to approximate the
solution fields of the classical vortex shedding benchmark (flow over cylinder).

This script operates in two modes:
   1) As the main program, to train the model.
   2) As an importable module, to use the trained model.

To train the model, open a terminal in the top-level `extrafeathers` directory, and:

  python -m demo.vae.main

The model will be saved in `saved_model_dir`, defined in config below.

The ELBO history, and some visualizations from the training process, will also be saved.

Note that some visualizations are created separately by running `anim.py` after the model
has been trained:

 - The evolution of the encoding of the dataset.
 - Animations.

To load the trained model, in an IPython session:

  import demo.vae.main as main
  main.model.my_load("demo/output/vae/model/final")  # or whatever

Now you can e.g.:

  import matplotlib.pyplot as plt
  import demo.vae.plotter as plotter
  plotter.plot_latent_image(21)
  plt.show()

Be aware that TensorFlow can be finicky with regard to resetting, if you wish to plot
different model instances during the same session. As of this writing (January 2023,
tf 2.12-nightly), to reset the model, you'll need something like:

  import gc
  import importlib
  import tensorflow as tf
  importlib.reload(main)  # re-instantiate CVAE
  tf.keras.backend.clear_session()  # clean up dangling tensors
  gc.collect()  # and make sure they are deleted

At this point, you should be able to load another trained model instance and it should work.

The implementation is based on combining material from these two tutorials:
  https://www.tensorflow.org/tutorials/generative/cvae
  https://keras.io/examples/generative/vae/

References:

  Diederik P. Kingma and Max Welling. 2019. An introduction to variational autoencoders.
  https://arxiv.org/abs/1906.02691

  Gabriel Loaiza-Ganem and John P. Cunningham. 2019. The continuous Bernoulli: fixing a pervasive error
  in variational autoencoders.
  https://arxiv.org/abs/1907.06845

  Shuyu Lin, Stephen Roberts, Niki Trigoni, and Ronald Clark. 2019. Balancing reconstruction quality
  and regularisation in evidence lower bound for variational autoencoders.
  https://arxiv.org/abs/1909.03765

  Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. MIT press.
  http://www.deeplearningbook.org
"""

# The REPL server allows inspecting/saving anything accessible from module-global scope
# while the process is live. To connect, `python -m unpythonic.net.client localhost`.
import unpythonic.net.server as repl_server

# TODO: in dataset overlay, vary marker size by the variance of each codepoint
#       (Need to compute the local scaling factor introduced by the coordinate
#        transformation in the figure, and choose a global overall scaling so
#        that it looks good. For the local scaling factor in quantile mode,
#        a finite difference of the interpolation grid is good enough.)

# TODO: use an early-stopping criterion to avoid overfitting the training set?
#
# TODO: conform better to the Keras OOP API (right now we have a Model that doesn't behave as the is-a implies)
#  - for a custom Keras object, `train_step` should be a method, not a separate function
#  - `call` (although somewhat useless for an autoencoder) should be implemented
#  - how to support `fit` and `predict`?
#  - saving/serialization with standard API
# TODO: register some monitored metrics? This would allow using the `EarlyStopping` class from the Keras API.
# TODO: implement metrics from "Early stopping but when?" paper

# TODO: For use with PDE solution fields:
#   - Project the PDE solution to a uniform grid (uniform, since we use a convolutional NN)
#   - Change the decoder model to a Gaussian (with learnable variance), as suggested in the paper by Lin et al.
#   - Or better yet, first check the distribution of the actual fields (observed data!)

import sys

from unpythonic import ETAEstimator, timer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from extrafeathers import plotmagic

from .config import (latent_dim,
                     output_dir, fig_format,
                     test_sample_fig_basename,
                     latent_space_fig_basename,
                     elbo_fig_filename)
from .cvae import CVAE, train_step, compute_loss
from .plotter import (plot_test_sample_image,
                      plot_elbo,
                      plot_latent_image)
from .util import clear_and_create_directory, preprocess_images

# --------------------------------------------------------------------------------
# Training config

# batch_size = 32  # for CPU
# batch_size = 128  # faster on GPU, still acceptable generalization on discrete Bernoulli
batch_size = 64  # acceptable generalization on continuous Bernoulli

# For this particular model, the test set ELBO saturates somewhere between epoch 100...200.
n_epochs = 200

# For a discussion of NN optimization methods, see the Deep Learning book by Goodfellow et al.
optimizer = tf.keras.optimizers.Adam(1e-4)

# --------------------------------------------------------------------------------
# Main program - model training

# For overlay plotting, it's convenient to have these here.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

model = CVAE(latent_dim)

def main():
    # Make preparations

    clear_and_create_directory(output_dir)

    # Set up figures
    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 4))  # test example
    fig1.tight_layout()
    fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))  # ELBO history
    fig2.tight_layout()
    fig3, axs3 = plt.subplots(1, 1, figsize=(10, 10))  # latent space
    fig3.tight_layout()

    # must call `plt.show` once before `plotmagic.pause` works
    plt.show()
    plt.draw()
    plotmagic.pause(0.001)

    train_size = train_images.shape[0]  # 60k
    test_size = test_images.shape[0]  # 10k

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

    # Train the model

    num_examples_to_generate = 16

    # Keeping the random vector constant for generation (prediction), it will be easier to see the improvement.
    # random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # Or... we can pick a sample of the test set for generating the corresponding reconstructed images.
    assert num_examples_to_generate <= batch_size
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    # Debug / info
    model.encoder.summary()
    model.decoder.summary()

    # Plot the random initial state
    plot_test_sample_image(test_sample, epoch=0, figno=1)
    plot_test_sample_image(test_sample, epoch=0, figno=1)  # and again to prevent axes crawling
    fig1.savefig(f"{output_dir}{test_sample_fig_basename}_0000.{fig_format}")
    fig1.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

    e = plot_latent_image(21, figno=3, epoch=0)
    e = plot_latent_image(21, figno=3, epoch=0)  # and again to prevent axes crawling
    fig3.savefig(f"{output_dir}{latent_space_fig_basename}_0000.{fig_format}")
    fig3.canvas.draw_idle()

    # Train the model
    est = ETAEstimator(n_epochs, keep_last=10)
    train_elbos = []
    test_elbos = []
    with timer() as tim_total:
        for epoch in range(1, n_epochs + 1):
            # SGD using one pass through the training set (with the batches set up previously)
            with timer() as tim_train:
                running_mean = tf.keras.metrics.Mean()
                for train_x in train_dataset:
                    running_mean(train_step(model, train_x, optimizer))
                train_elbo = -running_mean.result()
                train_elbos.append(train_elbo)

            # Performance estimation: ELBO on the test set
            with timer() as tim_test:
                running_mean = tf.keras.metrics.Mean()
                for test_x in test_dataset:
                    running_mean(compute_loss(model, test_x))
                test_elbo = -running_mean.result()
                test_elbos.append(test_elbo)

            # Plot the progress
            with timer() as tim_plot:
                # Test sample
                plot_test_sample_image(test_sample, epoch=epoch, figno=1)
                fig1.savefig(f"{output_dir}{test_sample_fig_basename}_{epoch:04d}.{fig_format}")
                fig1.canvas.draw_idle()

                # ELBO
                epochs = np.arange(1, epoch + 1)
                plot_elbo(epochs, train_elbos, test_elbos, epoch=epoch, figno=2)
                fig2.savefig(f"{output_dir}{elbo_fig_filename}.{fig_format}")
                fig2.canvas.draw_idle()

                # Latent space
                e = plot_latent_image(21, figno=3, epoch=epoch)
                fig3.savefig(f"{output_dir}{latent_space_fig_basename}_{epoch:04d}.{fig_format}")
                fig3.canvas.draw_idle()
                # overlay_datapoints(train_images, train_labels, e)  # out of memory on GPU

            # Save current model coefficients, and the ELBO history so far
            with timer() as tim_save:
                model.my_save(f"{output_dir}model/{epoch:04d}")
                np.savez(f"{output_dir}elbo.npz", epochs=epochs, train_elbos=train_elbos, test_elbos=test_elbos)

            est.tick()
            # dt_avg = sum(est.que) / len(est.que)
            print(f"Epoch: {epoch}, training set ELBO {train_elbo:0.6g}: test set ELBO {test_elbo:0.6g}, epoch walltime training {tim_train.dt:0.3g}s, testing {tim_test.dt:0.3g}s, plotting {tim_plot.dt:0.3g}s, saving {tim_save.dt:0.3g}s; {est.formatted_eta}")
    print(f"Total model training wall time: {tim_total.dt:0.6g}s")

    # # Save the trained model.
    # # TODO: Saving a CVAE instance using the official Keras serialization API doesn't work yet.
    # # force the model to build its graph to make it savable
    # dummy_data = tf.random.uniform((batch_size, 28, 28, 1))
    # _ = model(dummy_data)
    # model.save("my_model")
    #
    # this custom saving hack (saving the encoder/decoder separately) works
    model.my_save(f"{output_dir}model/final")

    # Visualize final state
    plot_test_sample_image(test_sample, figno=1)
    fig1.savefig(f"{output_dir}{test_sample_fig_basename}_final.{fig_format}")
    fig1.canvas.draw_idle()
    e = plot_latent_image(21, figno=3)  # noqa: F841
    fig3.savefig(f"{output_dir}{latent_space_fig_basename}_final.{fig_format}")
    fig3.canvas.draw_idle()

    # # ...and once again with a training dataset overlay
    # # out of memory on GPU, let's not do this in training
    # overlay_datapoints(train_images, train_labels, e)
    # fig3.savefig(f"{output_dir}{latent_space_fig_basename}_annotated.{fig_format}")
    # fig3.canvas.draw_idle()

if __name__ == "__main__":
    # HACK: If `main.py` is running as main program, but we also import it somewhere, then there will be
    # two independent copies, and the plotter will get the wrong one when fetching the default model
    # instance during training. So let's overwrite the other copy. Then the import system will see it as
    # already imported, and just return the existing instance.
    #
    # We only need this hack because the default model instance lives in `main.py`, which is also the
    # main program for training the model.
    sys.modules["demo.vae.main"] = sys.modules["__main__"]

    # To allow easy access to our global-scope variables in the live REPL session,
    # we make the main module (this module) available as `main` in the REPL scope.
    import sys
    repl_server.start(locals={"main": sys.modules["__main__"]})

    plt.ion()
    main()
    plt.ioff()
    plt.show()
