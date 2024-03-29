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

With the MNIST digits dataset, the ELBO should quickly rise to ~1300 (on model variant 7 and later)
after ~10 epochs if the random init is good.

Some variants can be finicky with the initialization. Just try again, if:

  - For model variant 8 and older:
    - The loss becomes NaN. In this case, training automatically stops after the epoch completes.
    - The latent space goes entirely black or entirely white near the start of the training,
      and the ELBO value seems stuck (usually near 700 ... 900).
  - For model variant 9:
    - Some pixels get stuck entirely white, and the first few periods of higher learning rate
      fail to shake them loose.

Model variants 8 and later are less susceptible to bad random inits. These variants incorporate
instance normalization after each set of ResNet layers, and also a dropout regularizer to improve
generalization.

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

Many more can be found in more specific docstrings, and in the source code comments.
"""

# The REPL server allows inspecting/saving anything accessible from module-global scope
# while the process is live. To connect, `python -m unpythonic.net.client localhost`.
import unpythonic.net.server as repl_server

# TODO: Conform better to the Keras OOP API. Anything left to do?
#
#  - `model.fit` (train), `model.evaluate` (test/validate), `model.predict` (infer, full roundtrip through the AE) and `model.save` work as expected.
#    - NOTE: CVAE uses custom training and test steps, and also the training script (`main.py`) is customized to produce specific visualizations while training.
#      While we have attempted to support the standard API, some advanced features might not work.
#
#  - The `training` flag of `call` is now supported.
#    - For an autoencoder, `__call__` is basically useless, except for validation, and as documentation of a full round-trip through the AE.
#      The custom training step of a VAE certainly never uses `__call__`, as computing the ELBO loss requires access to not only `xhat`,
#      but also to the code point `z`.
#    - The whole point of an AE is that although the encoder and decoder are trained jointly, they are separate submodels. Being able to invoke
#      the submodels separately gives us access to the latent representation, which encodes useful high-level features extracted from the training data.
#    - For the submodels, `__call__` is very useful; use `model.encoder(x)` and `model.decoder(z)` as appropriate.
#
#  - Division of responsibilities in Keras between the `__call__` operator and the `call` method:
#    - `call` is where to implement a user override; the `__call__` operator is for actually calling the model (it internally invokes `call`, plus does extra stuff).
#      https://stackoverflow.com/questions/57103604/why-keras-use-call-instead-of-call
#      https://www.tensorflow.org/api_docs/python/tf/keras/Model#call
#  - The `training` kwarg of `call`:
#    - Mode control: training or inference. Some layers (e.g. `Dropout`, `BatchNormalization`) behave differently at inference vs. training time.
#      https://keras.io/getting_started/faq/#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute
#    - So when calling any NN (in our case, encoder or decoder submodel) manually, we must pass the `training` kwarg.

# TODO: Use an early-stopping criterion to avoid overfitting the training set?
# TODO: `EarlyStopping` class from the Keras API. Do we need to register some more metrics to use it, or is just the loss enough?
#  We should register a validation loss?
#  https://keras.io/getting_started/faq/#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore

# TODO: Gradient accumulation for really large batches?
#   - Larger batch size is generally better, because the average (mean) gradient over the training data better reflects the behavior of the data distribution
#     than the gradient based only on a subset of the training data.
#     - One caveat (Keskar et al., 2017): numerically, this tends to make the optimizer behave steep minima, which hurts generalization.
#       https://arxiv.org/abs/1609.04836
#   - By default, the Keras optimizers make one gradient descent step per batch.
#   - Gradient accumulation and averaging over multiple batches (before performing a gradient descent step) would allow using very large batch sizes
#     where the whole batch does not fit into VRAM simultaneously, by processing it in several sub-batches (each small enough to fit into VRAM).
#   - Keras itself does not support gradient accumulation, but there is a wrapper add-on:
#     https://github.com/run-ai/runai/tree/master/runai/ga
#     How this interacts with the loss-scaling wrapper is an open question (which order should we nest them in?).

# TODO: For handling experimental data, add denoising, see e.g. https://davidstutz.de/denoising-variational-auto-encoders/
#   - The implementation is a small modification to the ELBO objective, but the theory behind it is significantly different from the classical VAE.
#   - Simulation-based data is clean, shouldn't need a denoiser.

# TODO: API: explore how we could implement more of the stochastics via `tensorflow_probability`, could be cleaner.

# TODO: For use with PDE solution fields:
#   - Project the PDE solution to a uniform grid (uniform, since we use a convolutional NN)
#   - Change the decoder model to a Gaussian (with learnable variance), as suggested in the paper by Lin et al.
#   - Or better yet, first check the distribution of the actual fields (observed data!)
#   - Encode extra parameters (e.g. fluid viscosity) as extra channels in the data?
#     - Better: have an extra input to the encoder, and then concatenate it to the final Dense layer.
#   - Estimate coverage of data manifold, e.g. using FID or precision-and-recall (see the latent diffusion
#     paper by Rombach et al., 2022)
#       https://arxiv.org/abs/2112.10752
#   - Other performance metrics: KL divergence of posterior from prior; mutual information between `x` and `z`; number of code dimensions actually in use.
#     See Dieng et al. (2019):
#       https://arxiv.org/abs/1807.04863

# TODO: Just for the lulz, we could try implementing a classifier on top of this.
#  - Map all training samples through the trained encoder. Save the resulting code points, with corresponding input labels.
#  - To classify a new data point, encode it. Use the labels of nearby training code points to predict what the class should be.
#  - E.g., form a confidence score for each class, based on an inverse-distance-weighted proportion of each training label within some configurable radius r.

from collections import defaultdict
import shutil
import sys

from unpythonic import ETAEstimator, timer
from unpythonic.env import env

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from extrafeathers import plotmagic

from .config import (latent_dim,
                     output_dir, fig_format,
                     test_sample_fig_basename,
                     latent_space_fig_basename,
                     elbo_fig_filename)
from . import clr
from .cvae import CVAE
from .plotter import (find_adversarial_samples,
                      plot_test_sample_image,
                      plot_elbo,
                      plot_latent_image,
                      plot_manifold)
from .util import clear_and_create_directory, preprocess_images

# --------------------------------------------------------------------------------
# Training config

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# # DEBUG TESTING
# variant = 0
# n_epochs = 10

# Choose the model variant. See `cvae.py`.
variant = 11

# test sample visualization
n_per_class = 40
test_sample_columns = 20
test_sample_zoom = 0.5

# When `latent_dim >= 3`, how many test samples to use to visualize the learned manifold.
manifold_samples = 4000

# We use the standard definition of "epoch": an epoch is one full pass over the training data set.
n_epochs = 200

# TODO: Optimal batch size?
#   Approximately same quality of training for `total number of gradient updates = constant`?
#     updates/epoch = data size / batch size
#     total updates = epochs * updates/epoch = epochs * data size / batch size
#   Thus, for the same data, and all else fixed, keep the ratio `epochs / batch size` constant to keep the training result quality constant?
#
# batch_size = 32  # CPU
if variant <= 7:
    batch_size = 1024  # 6GB VRAM, fp16, model variant 7; optimal training speed on RTX Quadro 3000 Mobile?
else:
    batch_size = 512  # 6GB VRAM, fp32, model variant 7; or fp16, model variant ≥ 8

# Choose dtype policy (which is best depends on your device)
#   https://tensorflow.org/guide/mixed_precision
# policy = tf.keras.mixed_precision.Policy("float32")
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
with env() as e:  # avoid polluting top-level scope with temporaries
    e.d, e.m = divmod(train_images.shape[0], batch_size)
    steps_per_epoch = e.d + int(e.m > 0)  # last one for leftovers (if number of training samples not divisible by batch size)

# # Constant learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# # Exponentially decaying learning rate
# decay_epochs = 50  # In the exponential schedule, after each `decay_epochs`, the lr has reached `decay_rate` times its original value.
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
#                                                              decay_steps=decay_epochs * steps_per_epoch,
#                                                              decay_rate=0.25)

# Cyclical learning rate
# This accelerates convergence according to Smith (2017):
#   https://arxiv.org/abs/1506.01186
if variant == 9:
    # Model variant 9 needs a higher max LR, maybe because it uses many more dropout layers.
    INIT_LR, MAX_LR = 1e-4, 2e-2
elif variant in (10, 11):
    if latent_dim == 2:
        INIT_LR, MAX_LR = 2e-4, 1e-2
    else:  # tested with `latent_dim = 20`
        INIT_LR, MAX_LR = 2e-4, 2e-2
else:
    # Primarily tested with model variants 7 and 8.
    INIT_LR, MAX_LR = 1e-4, 2e-3
# "triangular2" schedule of Smith (2017)
# `step_size` = half cycle length, in optimizer steps; Smith recommends `(2 ... 10) * steps_per_epoch`
# lr_schedule = clr.Triangular2CyclicalLearningRate(lr0=INIT_LR,
#                                                   lr1=MAX_LR,
#                                                   half_cycle_length=10 * steps_per_epoch,
#                                                   cycle_profile="smooth")  # TODO: which profile is best?
lr_schedule = clr.ExponentialCyclicalLearningRate(lr0=INIT_LR,
                                                  lr1=MAX_LR,
                                                  gamma=0.9,
                                                  half_cycle_length=2 * steps_per_epoch,
                                                  cycle_profile="smooth")  # TODO: which profile is best?

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
# Optimizer = tf.keras.optimizers.SGD  # 5% faster than Adam, and produces clear cycles in the ELBO graphs, but converges slower, and the final result is not as good.
Optimizer = tf.keras.optimizers.Adam

# --------------------------------------------------------------------------------
# Main program - model training

print(f"{__name__}: Compute dtype: {policy.compute_dtype}")
print(f"{__name__}: Variable dtype: {policy.variable_dtype}")

model = CVAE(latent_dim=latent_dim, variant=variant)

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

    # Representative test sample
    fig1, axs1 = plt.subplots(1, 1, figsize=(test_sample_zoom * float(2 * test_sample_columns + 1),
                                             test_sample_zoom * float(test_sample_columns)))
    fig1.tight_layout()

    # ELBO history
    fig2, axs2 = plt.subplots(1, 1, figsize=(6, 4))
    fig2.tight_layout()

    # Latent space
    if latent_dim == 2:
        fig3, axs3 = plt.subplots(1, 1, figsize=(10, 10))  # latent space
    else:
        fig3, axs3 = plt.subplots(1, 1, figsize=(6, 5))  # learned manifold (while training, we'll be plotting in "fast" mode with t-SNE only)
    fig3.tight_layout()

    # Adversarial test sample (worst l2 error in pixel space)
    fig4, axs4 = plt.subplots(1, 1, figsize=(test_sample_zoom * float(2 * test_sample_columns + 1),
                                             test_sample_zoom * float(test_sample_columns)))
    fig4.tight_layout()

    # must call `plt.show` once before `plotmagic.pause` works
    plt.show()
    plt.draw()
    plotmagic.pause(0.001)

    train_size = train_images.shape[0]  # 60k
    test_size = test_images.shape[0]  # 10k

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

    # Train the model

    # Keeping the random vector constant for generation (prediction), it will be easier to see the improvement.
    # num_examples_to_generate = 16
    # random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # # Or... we can pick a sample of the test set for generating the corresponding reconstructed images (roundtrip through the VAE).
    # num_examples_to_generate = 16
    # assert num_examples_to_generate <= batch_size
    # for test_batch in test_dataset.take(1):
    #     test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    # Or... improving further, pick a sample containing `n` examples from each class.
    def to_stacked_array(data_dict, n_per_class):  # Helper function: {label0: [image0, ...], ...} -> tensor [N, ny, nx, 1]
        data_dict = {k: v[:n_per_class] for k, v in data_dict.items()}  # Cut away any extra examples from each class
        assert all(len(v) == n_per_class for v in data_dict.values())   # The data should have at least the desired number of examples of each class
        sorted_data = sorted(data_dict.items(), key=lambda kv: kv[0])        # Sort classes: 0s first, then 1s, ...
        images_by_class = [v[1] for v in sorted_data]                   # Drop class labels: {label0: [image0, ...], ...} -> [[image0, ...], ...]
        stacked_by_class = [tf.stack(v) for v in images_by_class]       # Stack within each class: [[ny, nx, 1], ...] -> [K, ny, nx, 1]
        batched = tf.concat(stacked_by_class, axis=0)                   # Merge the stacks into a single batch
        n_classes = len(data_dict)
        assert batched.shape[0] == n_classes * n_per_class
        return batched

    def prepare_test_sample():  # just a namespace to drop temporaries to the GC as early as possible (don't keep unnecessary copies of the test data in RAM)
        test_examples_dict = defaultdict(list)
        for label, image in zip(test_labels, test_images):
            test_examples_dict[label].append(image)
            counts_by_class = [len(v) for v in test_examples_dict.values()]
            if min(counts_by_class) >= n_per_class:
                break
        assert len(test_examples_dict) == 10  # MNIST digits data; should have found at least one example of each class
        test_sample = to_stacked_array(test_examples_dict, n_per_class)
        return test_sample
    test_sample = prepare_test_sample()

    # # Recipe for finding dict keys from an unknown TF dataset:
    # # https://stackoverflow.com/questions/48825785/how-can-i-filter-tf-data-dataset-by-specific-values
    # import json
    # from google.protobuf.json_format import MessageToJson
    # for raw_record in no_idea_dataset.take(1):
    #     example = tf.train.Example()
    #     example.ParseFromString(raw_record.numpy())
    #     ##print(example) ##if image it will be toooolong
    #     m = json.loads(MessageToJson(example))
    #     print(m['features']['feature'].keys())

    # Debug / info
    model.encoder.summary()
    model.decoder.summary()

    def plot_adversarial_sample_image(epoch):
        # Sorted by class (easier to see which classes the CVAE struggles with)
        print("Finding test samples with highest pixel-space l2 error...")
        with timer() as tim:
            total_l2_error, ks_by_l2_error = find_adversarial_samples(test_images, test_labels)  # yes, all 10k of them!
        print(f"    Done in {tim.dt:0.6g}s.")
        # prepare the adversarial samples in the same format as in `prepare_test_sample`
        test_examples_dict = {k: test_images[v, :, :, :] for k, v in ks_by_l2_error.items()}
        assert len(test_examples_dict) == 10  # MNIST digits data; should have found at least one example of each class
        test_sample = to_stacked_array(test_examples_dict, n_per_class)
        mean_l2_error = total_l2_error / test_images.shape[0]  # for the whole test data (not just the adversarial samples)
        plot_test_sample_image(test_sample,
                               custom_title=f"Test samples with highest pixel-space l2 error (per class), test set mean l2 error {mean_l2_error:0.6g}",
                               epoch=epoch, figno=4, cols=test_sample_columns, zoom=test_sample_zoom)

        # # Not sorted by class
        # total_l2_error, ks_by_l2_error = find_adversarial_samples(test_images)
        # n = test_sample.shape[0]  # plot the same total number of samples as in figure 1
        # test_sample = tf.constant(test_images[ks_by_l2_error[:n], :, :, :])
        # mean_l2_error = total_l2_error / test_images.shape[0]  # for the whole test data (not just the adversarial samples)
        # plot_test_sample_image(test_sample,
        #                        custom_title=f"Test samples with highest pixel-space l2 error, test set mean l2 error {mean_l2_error:0.6g}",
        #                        epoch=epoch, figno=4, cols=test_sample_columns, zoom=test_sample_zoom)

    # Plot the random initial state
    plot_test_sample_image(test_sample,
                           custom_title="Representative test samples",
                           epoch=0, figno=1, cols=test_sample_columns, zoom=test_sample_zoom)
    plot_test_sample_image(test_sample,
                           custom_title="Representative test samples",
                           epoch=0, figno=1, cols=test_sample_columns, zoom=test_sample_zoom)  # and again to prevent axes crawling
    fig1.savefig(f"{output_dir}{test_sample_fig_basename}_0000.{fig_format}")
    fig1.canvas.draw_idle()   # see source of `plt.savefig`; need this if 'transparent=True' to reset colors

    plot_adversarial_sample_image(epoch=0)
    plot_adversarial_sample_image(epoch=0)  # and again to prevent axes crawling
    fig4.savefig(f"{output_dir}{test_sample_fig_basename}_worstl2_0000.{fig_format}")
    fig4.canvas.draw_idle()

    if latent_dim == 2:
        plot_latent_image(21, figno=3, epoch=0)
        plot_latent_image(21, figno=3, epoch=0)  # and again to prevent axes crawling
    else:
        plot_manifold(test_images[:manifold_samples, :, :, :], test_labels[:manifold_samples], methods="fast", figno=3, epoch=0)
        plot_manifold(test_images[:manifold_samples, :, :, :], test_labels[:manifold_samples], methods="fast", figno=3, epoch=0)  # and again to prevent axes crawling
    fig3.savefig(f"{output_dir}{latent_space_fig_basename}_0000.{fig_format}")
    fig3.canvas.draw_idle()

    # Train the model
    est = ETAEstimator(n_epochs, keep_last=10)
    learning_rates = []
    train_elbos = []
    test_elbos = []
    generalization_losses = []
    training_progresses = []
    best_epoch = 0
    with timer() as tim_total:
        for epoch in range(1, n_epochs + 1):
            prev_iterations = optimizer.iterations.numpy()

            # SGD (with Adam) using one pass through the training set (with the batches set up previously)
            with timer() as tim_train:
                history = model.fit(train_dataset, epochs=1)
                losses_by_epoch = history.history["loss"]
                train_loss = losses_by_epoch[0]  # we ran just one epoch (since we loop over epochs manually)
                if np.isnan(train_loss):
                    raise ValueError(f"Training loss became NaN at epoch {epoch}, stopping.")
                train_elbo = -train_loss
                train_elbos.append(train_elbo)

            # Performance estimation: ELBO on the test set (technically, used as a validation set)
            with timer() as tim_test:
                test_loss = model.evaluate(test_dataset)
                if np.isnan(test_loss):
                    raise ValueError(f"Test loss became NaN at epoch {epoch}, stopping.")
                test_elbo = -test_loss
                test_elbos.append(test_elbo)

            # Plot the progress
            with timer() as tim_plot:
                # Representative test sample
                plot_test_sample_image(test_sample,
                                       custom_title="Representative test samples",
                                       epoch=epoch, figno=1, cols=test_sample_columns, zoom=test_sample_zoom)
                fig1.savefig(f"{output_dir}{test_sample_fig_basename}_{epoch:04d}.{fig_format}")
                fig1.canvas.draw_idle()

                # Adversarial test sample
                plot_adversarial_sample_image(epoch=epoch)
                fig4.savefig(f"{output_dir}{test_sample_fig_basename}_worstl2_{epoch:04d}.{fig_format}")
                fig4.canvas.draw_idle()

                # ELBO
                epochs = np.arange(1, epoch + 1)
                optimizer_steps = np.arange(0, epoch * steps_per_epoch + 1)
                lr_epochs = optimizer_steps / steps_per_epoch  # optimizer total step number represented as a fractional epoch number
                lrs = lr_schedule(optimizer_steps)
                plot_elbo(epochs, train_elbos, test_elbos,
                          epoch=epoch,
                          lr_epochs=lr_epochs, lrs=lrs,
                          figno=2)
                fig2.savefig(f"{output_dir}{elbo_fig_filename}.{fig_format}")
                fig2.canvas.draw_idle()

                # Latent space
                if latent_dim == 2:
                    plot_latent_image(21, figno=3, epoch=epoch)
                    # overlay_datapoints(train_images, train_labels, e)  # very slow, let's not do it while training
                else:
                    plot_manifold(test_images[:manifold_samples, :, :, :], test_labels[:manifold_samples], methods="fast", figno=3, epoch=epoch)
                fig3.savefig(f"{output_dir}{latent_space_fig_basename}_{epoch:04d}.{fig_format}")
                fig3.canvas.draw_idle()

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
            #   https://www.researchgate.net/publication/2874749_Early_Stopping_-_But_When
            max_test_elbo = max(test_elbos)  # only running a couple hundred epochs; O(n) not a problem.
            generalization_loss = 1.0 - test_elbo / max_test_elbo
            generalization_losses.append(generalization_loss)
            if generalization_loss == 0.0:
                best_epoch = epoch

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
            total_dt = tim_train.dt + tim_test.dt + tim_plot.dt + tim_save.dt
            print(f"Epoch: {epoch} [best {best_epoch} @ ELBO {max_test_elbo:0.6g}], LR {prev_learning_rate:0.6g} ... {learning_rate:0.6g}, ELBO train {train_elbo:0.6g}, test {test_elbo:0.6g}; GL {generalization_loss:0.6g}, P5 {training_progress:0.6g}; opt. steps {total_iterations} (this epoch {epoch_iterations}).\nEpoch walltime {total_dt:0.3g}s (train {tim_train.dt:0.3g}s, test {tim_test.dt:0.3g}s, plot {tim_plot.dt:0.3g}s, save {tim_save.dt:0.3g}s); {est.formatted_eta}")
    print(f"Total wall time for training run: {tim_total.dt:0.6g}s")

    # Save the trained model. (Just copy the best epoch, already saved during the training loop.)
    shutil.copy2(src=f"{output_dir}model/{best_epoch:04d}.keras",
                 dst=f"{output_dir}model/final.keras")
    # Touch a file, with the filename telling the user which epoch was the best one
    descriptive_filename = f"{output_dir}00_best_is_epoch_{best_epoch}_with_ELBO_{round(max_test_elbo)}.txt"
    with open(descriptive_filename, "wt"):
        pass

    # Visualize final state
    shutil.copy2(src=f"{output_dir}{test_sample_fig_basename}_{best_epoch:04d}.{fig_format}",
                 dst=f"{output_dir}{test_sample_fig_basename}_final.{fig_format}")
    shutil.copy2(src=f"{output_dir}{latent_space_fig_basename}_{best_epoch:04d}.{fig_format}",
                 dst=f"{output_dir}{latent_space_fig_basename}_final.{fig_format}")
    shutil.copy2(src=f"{output_dir}{test_sample_fig_basename}_worstl2_{best_epoch:04d}.{fig_format}",
                 dst=f"{output_dir}{test_sample_fig_basename}_worstl2_final.{fig_format}")

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
