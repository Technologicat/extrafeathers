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

The random initialization of the model can be finicky. With the MNIST digits dataset,
the ELBO should rise to ~1300 (on model variant=7) after a few epochs if the init is good.
Just try again, if:

  - The loss becomes NaN, which crashes the training.
  - The latent space goes entirely black or entirely white near the start of the training,
    and the ELBO value seems stuck.

The model will be saved in `<output_dir>/model/XXXX.keras`, where `output_dir` is defined in `config.py`,
and XXXX is either a four-digit epoch number starting from 0001, or "final" for the final result of
a completed training run.

The ELBO history, and some visualizations from the training process, will also be saved in `output_dir`.

Note that some visualizations are created separately by running `anim.py` after the model
has been trained:

 - The evolution of the encoding of the dataset.
 - Animations.

How to load the trained model in an IPython session: see `test_script.py`.
The script also makes a datapoint overlay plot just for the final state.

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

# TODO: Conform better to the Keras OOP API
#  - `fit`, `evaluate` and `save` now work as expected!
#  - what does `predict` do, how to support it? (How does it differ from `call`?)
#  - how does the `__call__` operator differ from the `call` method? How are the responsibilities divided between these in Keras?
#  - what should happen when the model is called (`model(...)`); what should the `training=False` kwarg do?

# TODO: Use an early-stopping criterion to avoid overfitting the training set?
# TODO: `EarlyStopping` class from the Keras API. Do we need to register some more metrics to use it, or is just the loss enough?

# TODO: For `latent_dim > 2`, in the plotter, add a second dimension reduction step to compress into 2d. Process the z space via e.g. diffusion maps or t-SNE.

# TODO: Add denoising, see e.g. https://davidstutz.de/denoising-variational-auto-encoders/

# TODO: API: explore how we could implement more of the stochastics via `tensorflow_probability`, could be cleaner.

# TODO: Explore if we can set up skip-connections from an encoder layer to the decoder layer of the same size.
#   - This kind of architecture is commonly seen in autoencoder designs used for approximating PDE solutions
#     (PINNs, physically informed neural networks); it should speed up training of the encoder/decoder combination
#     by allowing the decoded output to directly influence the weights on the encoder side.
#   - Using the functional API, we should be able to set up three `Model`s that share layer instances:
#     the encoder, the decoder, and (for training) the full autoencoder that combines both and adds the
#     skip-connections from the encoder side to the decoder side.
#   - But how to run such a network in decoder-only mode (i.e. generative mode), where the encoder layers are not available?
#     For real data, we could save the encoder layer states as part of the coded representation. But for generating new data, that's a no-go.

# TODO: For use with PDE solution fields:
#   - Project the PDE solution to a uniform grid (uniform, since we use a convolutional NN)
#   - Change the decoder model to a Gaussian (with learnable variance), as suggested in the paper by Lin et al.
#   - Or better yet, first check the distribution of the actual fields (observed data!)
#   - Encode extra parameters (e.g. fluid viscosity) as extra channels in the data?
#     - Better: have an extra input to the encoder, and then concatenate it to the final Dense layer.
#   - Estimate coverage of data manifold, e.g. using FID or precision-and-recall (see the latent diffusion
#     paper by Rombach et al., 2022)

# TODO: Just for the lulz, we could try implementing a classifier on top of this.
#  - Map all training samples through the trained encoder. Save the resulting code points, with corresponding input labels.
#  - To classify a new data point, encode it. Use the labels of nearby training code points to predict what the class should be.
#  - E.g., form a confidence score for each class, based on an inverse-distance-weighted proportion of each training label within some configurable radius r.

import sys

from unpythonic import ETAEstimator, timer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa

from extrafeathers import plotmagic

from .config import (latent_dim,
                     output_dir, fig_format,
                     test_sample_fig_basename,
                     latent_space_fig_basename,
                     elbo_fig_filename)
from .cvae import CVAE
from .plotter import (plot_test_sample_image,
                      plot_elbo,
                      plot_latent_image)
from .util import clear_and_create_directory, preprocess_images

# --------------------------------------------------------------------------------
# Training config

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# We use the standard definition of "epoch": an epoch is one full pass over the training data set.
n_epochs = 140

# TODO: Optimal batch size?
#   Approximately same quality of training for `total number of gradient updates = constant`?
#     updates/epoch = data size / batch size
#     total updates = epochs * updates/epoch = epochs * data size / batch size
#   Thus, for the same data, and all else fixed, keep the ratio `epochs / batch size` constant to keep the training result quality constant?
#
# batch_size = 32  # CPU
batch_size = 1024  # 6GB VRAM, fp16, model variant=7; optimal training speed on RTX Quadro 3000 Mobile?
# batch_size = 512  # 6GB VRAM, fp32, model variant=7
# batch_size = 256
# batch_size = 128
# batch_size = 64

# Choose dtype policy (which is best depends on your device)
#   https://tensorflow.org/guide/mixed_precision
# policy = tf.keras.mixed_precision.Policy("float32")  # CPU
# policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")  # RTX 3xxx and later should have the tensor core hardware to accelerate bf16
policy = tf.keras.mixed_precision.Policy("mixed_float16")  # Quadro 3000 (based on RTX 2xxx chip)
tf.keras.mixed_precision.set_global_policy(policy)

# Set up the optimizer.
#
# For a general discussion of NN optimization methods, explaining many of the popular algorithms,
# see the Deep Learning book by Goodfellow et al. (2016).
#
# A Keras `Optimizer` increments `self.iterations` by one every time its `apply_gradients` method is called.
#   https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
#
# One optimizer step processes one batch of input data. Hence, optimizer steps per epoch is:
d, m = divmod(train_images.shape[0], batch_size)
steps_per_epoch = d + int(m > 0)  # last one for leftovers (if number of training samples not divisible by batch size)

# # Constant learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# # Exponentially decaying learning rate
# decay_epochs = 50  # In the exponential schedule, after each `decay_epochs`, the lr has reached `decay_rate` times its original value.
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
#                                                              decay_steps=decay_epochs * steps_per_epoch,
#                                                              decay_rate=0.25)

# Cyclical learning rate
# This accelerates convergence according to Smith (2015):
#   https://arxiv.org/abs/1506.01186
#
# TODO: `tensorflow_addons` is deprecated but there seems to be no official replacement for `CyclicalLearningRate` and friends. Make a local copy of the class?
#   https://www.tensorflow.org/addons/tutorials/optimizers_cyclicallearningrate
#   https://github.com/tensorflow/addons/issues/2807
#
# "triangular2" schedule of Smith (2015)
# `step_size` = half cycle length, in optimizer steps
INIT_LR, MAX_LR = 1e-4, 1e-2
lr_schedule = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                                             maximal_learning_rate=MAX_LR,
                                                             step_size=10 * steps_per_epoch)

# # learning schedule DEBUG
# steps = np.arange(0, n_epochs * steps_per_epoch)
# lr = lr_schedule(steps)
# plt.plot(steps / steps_per_epoch, lr)
# plt.xlabel("epoch")
# plt.ylabel("lr")
# plt.grid(visible=True, which="both")
# plt.axis("tight")
# plt.show()
# from sys import exit
# exit(0)

# Choose the optimizer algorithm. This can be any Keras optimizer that takes in a `learning_rate` kwarg.
Optimizer = tf.keras.optimizers.Adam

# --------------------------------------------------------------------------------
# Main program - model training

print(f"{__name__}: Compute dtype: {policy.compute_dtype}")
print(f"{__name__}: Variable dtype: {policy.variable_dtype}")

model = CVAE(latent_dim=latent_dim, variant=7)

optimizer = Optimizer(learning_rate=lr_schedule)
if policy.compute_dtype == "float16":  # mitigate gradient underflow with fp16
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
model.compile(optimizer=optimizer)  # pass only the optimizer here; the CVAE model implements a custom training step, and handles metrics and losses explicitly.

model.build((batch_size, 28, 28, 1))  # force the model to build its graph so that `model.save` works.

def main():
    # Make preparations

    clear_and_create_directory(output_dir)
    clear_and_create_directory(f"{output_dir}model")

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
    learning_rates = []
    train_elbos = []
    test_elbos = []
    generalization_losses = []
    training_progresses = []
    with timer() as tim_total:
        for epoch in range(1, n_epochs + 1):
            prev_iterations = optimizer.iterations.numpy()

            # SGD (with Adam) using one pass through the training set (with the batches set up previously)
            with timer() as tim_train:
                history = model.fit(train_dataset, epochs=1)
                losses_by_epoch = history.history["loss"]
                train_loss = losses_by_epoch[0]  # we ran just one epoch (since we loop over epochs manually)
                train_elbo = -train_loss
                train_elbos.append(train_elbo)

            # Performance estimation: ELBO on the test set (technically, used as a validation set)
            with timer() as tim_test:
                test_loss = model.evaluate(test_dataset)
                test_elbo = -test_loss
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

            # Store current learning rate, for visualization/debugging of the learning schedule
            prev_learning_rate = lr_schedule(prev_iterations)
            learning_rate = lr_schedule(optimizer.iterations)
            learning_rates.append(learning_rate)
            total_iterations = optimizer.iterations.numpy()
            epoch_iterations = total_iterations - prev_iterations

            # Estimate generalization quality and training progress.
            #
            # Similar to GL(t) in Prechelt (2000): "Early stopping, but when?"; but our ELBOs are positive
            # (because continuous distribution p(x)), which makes our loss function negative, so we adapt
            # the definition slightly.
            max_test_elbo = max(test_elbos)  # only running a couple hundred epochs; O(n) not a problem.
            generalization_loss = 1.0 - test_elbo / max_test_elbo
            generalization_losses.append(generalization_loss)

            # Similar to Pk(t) in Prechelt (2000).
            k = 5
            last_k_train_elbos = train_elbos[-k:]
            max_of_last_k_train_elbos = max(last_k_train_elbos)
            mean_of_last_k_train_elbos = sum(last_k_train_elbos) / len(last_k_train_elbos)
            training_progress = 1.0 - mean_of_last_k_train_elbos / max_of_last_k_train_elbos
            training_progresses.append(training_progress)

            # Save the current model coefficients, and some statistics
            with timer() as tim_save:
                # Use the ".keras" filename extension to save in the new format.
                #   https://www.tensorflow.org/tutorials/keras/save_and_load
                model.save(f"{output_dir}model/{epoch:04d}.keras", save_format="keras_v3")
                # model.my_save(f"{output_dir}model/{epoch:04d}")

                np.savez(f"{output_dir}elbo.npz",
                         epochs=epochs,
                         train_elbos=train_elbos,
                         test_elbos=test_elbos,
                         optimizer_iterations=total_iterations,
                         learning_rates=learning_rates,
                         generalization_losses=generalization_losses,
                         training_progresses=training_progresses,
                         batch_size=batch_size,
                         policy=policy.name)

            est.tick()
            # dt_avg = sum(est.que) / len(est.que)
            print(f"Epoch: {epoch}, LR {prev_learning_rate:0.6g} ... {learning_rate:0.6g}, ELBO train {train_elbo:0.6g}, test {test_elbo:0.6g}; GL(t) {generalization_loss:0.6g}, P5(t) {training_progress:0.6g}; opt. iter. {total_iterations} (this epoch {epoch_iterations}).\nEpoch walltime training {tim_train.dt:0.3g}s, testing {tim_test.dt:0.3g}s, plotting {tim_plot.dt:0.3g}s, saving {tim_save.dt:0.3g}s; {est.formatted_eta}")
    print(f"Total wall time for training run: {tim_total.dt:0.6g}s")

    # Save the trained model.
    model.save(f"{output_dir}model/final.keras", save_format="keras_v3")
    # # legacy custom saving hack (saving the encoder/decoder separately)
    # model.my_save(f"{output_dir}model/final")

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

    print("Model training complete.")


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
