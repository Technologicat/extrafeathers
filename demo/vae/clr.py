"""Cyclical learning rate.

Based on `tensorflow_addons.optimizers.cyclical_learning_rate`,
used under the Apache License 2.0.

This code is mirrored locally, because `tensorflow_addons` is deprecated (EOL May 2024)
and there is no official package with a replacement for `CyclicalLearningRate`.

This code has been modified slightly, for clarity and simplicity:

  - `cycle_mode` argument removed; the scaling function always scales by the cycle number.
  - Clearer names for arguments.
  - Learning rates no longer support callables.
  - Demystified computation of the `x` value.

We have also added a variant that uses infinitely smooth transitions; see the `cycle_profile` parameter.

See Smith (2017): Cyclical Learning Rates for Training Neural Networks.
  https://arxiv.org/abs/1506.01186
"""

__all__ = ["CyclicalLearningRate",
           "Triangular2CyclicalLearningRate",
           "ExponentialCyclicalLearningRate"]

from typing import Optional, Union, Callable

import numpy as np

import tensorflow as tf

from typeguard import typechecked

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]

# --------------------------------------------------------------------------------

# TODO: We already have a NumPy function of this in `extrafeathers.pdes.numutil`, but here we need a TensorFlow implementation.
def ψ(x, m=1.0, eps=0.01):
    """Building block for non-analytic smooth functions.

        ψ(x, m) := exp(-1 / x^m) χ(0, ∞)(x)

    where χ is the indicator function (1 if x is in the set, 0 otherwise).

    Prevents divide by zero by using a small epsilon.

    This is the helper function used in the construction of the standard
    mollifier.
    """
    cut_x = tf.where(tf.less(x, eps), eps * tf.ones_like(x), x)
    return tf.where(tf.less(x, eps), tf.zeros_like(x), tf.exp(-1.0 / cut_x**m))

def nonanalytic_smooth_transition(x, m=1.0):
    """Non-analytic smooth transition from 0 to 1, on interval x ∈ [0, 1].

    The transition is reflection-symmetric through the point (1/2, 1/2).

    Outside the interval:
        s(x, m) = 0  for x < 0
        s(x, m) = 1  for x > 1

    The parameter `m` controls the steepness of the transition region.
    Larger `m` packs the transition closer to `x = 1/2`, making it
    more abrupt (although technically, still infinitely smooth).

    `m` is passed to `ψ`, which see.
    """
    p = ψ(x, m)
    return p / (p + ψ(1 - x, m))

# --------------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="ExtraFeathers")
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A `LearningRateSchedule` for a cyclical LR with triangular cycles.

    You can pass this schedule directly into a
    `tf.keras.optimizers.legacy.Optimizer` as the learning rate.

        `lr0`: A scalar `float32` or `float64` `Tensor` or a Python number. The initial learning rate.
        `lr1`: A scalar `float32` or `float64` `Tensor` or a Python number. The maximum learning rate.
        `half_cycle_length`: A scalar `float32` or `float64` `Tensor` or a Python number.
                             How many optimizer steps (iterations) to reach `lr1`.
        `cycle_scale`: A single-argument function to scale the LR by the cycle number (1-based).

                       The default `None` means `lambda _: 1.0`, i.e., no scaling; this is the "triangular"
                       schedule of Smith.

                       Any callable can be passed, but if you want to serialize this object you
                       should only pass functions that are registered Keras serializables
                       (see `tf.keras.saving.register_keras_serializable` for more details).

        `cycle_profile`: One of:
           "linear": The original triangular sawtooth shape. C0 continuous at the seams.
           "cosine": Cosine shape. When `cycle_scale` is used, the profile is only C0 continuous
                     at the seams; all odd-order derivatives have a discontinuity at each seam.
           "smooth": An infinitely smooth (C∞ continuous) square-like shape.
                     This makes the LR vary smoothly as a function of the optimizer step number
                     for any `cycle_scale`, also at the seams. For this shape, derivatives of
                     all orders are zero at the seam.
        `name`: (optional) Name for the operation in the TensorFlow graph, for debugging.

    When the instance is called, the return value is the updated learning rate.
    """

    @typechecked
    def __init__(self,
                 lr0: FloatTensorLike,
                 lr1: FloatTensorLike,
                 half_cycle_length: FloatTensorLike,
                 cycle_scale: Optional[Callable] = None,
                 cycle_profile: str = "linear",
                 name: str = "CyclicalLearningRate"):
        if cycle_scale is None:
            cycle_scale = lambda _: 1.0
        if cycle_profile not in ("linear", "cosine", "smooth"):
            raise ValueError(f"Expected `cycle_profile` to be one of 'linear', 'cosine', 'smooth'; got '{cycle_profile}'")

        super().__init__()
        self.lr0 = lr0
        self.lr1 = lr1
        self.half_cycle_length = half_cycle_length
        self.cycle_scale = cycle_scale
        self.cycle_profile = cycle_profile
        self.name = name

    def __call__(self, step):
        """`step`: optimizer step number (running count from start of training)"""
        with tf.name_scope(self.name):
            # We convert these every time to account for possible changes by user code between calls.
            lr0 = tf.convert_to_tensor(self.lr0, name="lr0")
            dtype = lr0.dtype
            lr1 = tf.cast(self.lr1, dtype)
            half_cycle_length = tf.cast(self.half_cycle_length, dtype)

            step_as_dtype = tf.cast(step, dtype)

            cycle_length = 2 * half_cycle_length
            cycle_number = tf.floor(1 + step_as_dtype / cycle_length)

            # How this works:
            #   `1 + step_as_dtype / half_cycle_length`: 1-based number of current half-cycle,
            #                                            plus an offset in [0, 2] for the position
            #                                            within the current full cycle
            #   `2 * cycle_number`: how many half-cycles completed at the end of the current full cycle
            # Hence, before the `abs`, we have:
            #   cycle 1: [1, 3] - 2 = [-1, 1]
            #   cycle 2: [3, 5] - 4 = [-1, 1]
            #   ...
            # This expression goes from -1 to 1 within each cycle. So taking the `abs`:
            # `x` starts from 1, reaches 0 at the midpoint, and then increases back to 1.
            # This gives each cycle its triangular shape.
            x = tf.abs(1 + step_as_dtype / half_cycle_length - 2 * cycle_number)

            if self.cycle_profile == "cosine":
                x = (1 + tf.math.cos(np.pi * (1 + x))) / 2
            elif self.cycle_profile == "smooth":
                x = nonanalytic_smooth_transition(x, m=1.0)

            # The cycle shape is still upside down. Flipping it gives us the desired fraction of the LR delta
            # at the current optimizer step. We clip to ensure the LR always stays in [lr0, lr1] also numerically.
            y = tf.maximum(tf.cast(0, dtype), 1 - x)

            Δlr = lr1 - lr0
            return lr0 + y * Δlr * self.cycle_scale(cycle_number)

    def get_config(self):
        return {"lr0": self.lr0,
                "lr1": self.lr1,
                "half_cycle_length": self.half_cycle_length,
                "cycle_scale": self.cycle_scale,  # TODO: a callable might not be serializable (could be a lambda)
                "cycle_profile": self.cycle_profile,
                "name": self.name}


@tf.keras.utils.register_keras_serializable(package="ExtraFeathers")
class Triangular2CyclicalLearningRate(CyclicalLearningRate):
    """A `LearningRateSchedule` for a cyclical LR with decaying triangular cycles.

    This is the "triangular2" schedule of Smith.

    Arguments and return value as in `CyclicalLearningRate`.
    """
    @typechecked
    def __init__(self,
                 lr0: FloatTensorLike,
                 lr1: FloatTensorLike,
                 half_cycle_length: FloatTensorLike,
                 cycle_profile: str = "linear",
                 name: str = "Triangular2CyclicalLearningRate"):
        super().__init__(lr0=lr0,
                         lr1=lr1,
                         half_cycle_length=half_cycle_length,
                         cycle_scale=lambda cycle_number: 1 / (2.0 ** (cycle_number - 1)),
                         cycle_profile=cycle_profile,
                         name=name)

    def get_config(self):
        return {"lr0": self.lr0,
                "lr1": self.lr1,
                "half_cycle_length": self.half_cycle_length,
                "cycle_profile": self.cycle_profile,
                "name": self.name}


@tf.keras.utils.register_keras_serializable(package="ExtraFeathers")
class ExponentialCyclicalLearningRate(CyclicalLearningRate):
    """A `LearningRateSchedule` for a cyclical LR with decaying triangular cycles.

    "triangular2" is a special case of this schedule, with `gamma=0.5`.

    Differences to "exp_range" of Smith:

      - We do not decrease `lr0`; the decaying exponential scaling is applied
        to `Δlr := lr1 - lr0` instead.
      - We scale by cycle, nor by step number.

    Arguments and return value as in `CyclicalLearningRate`, except:

        `gamma`: Each cycle is scaled by `gamma ** (cycle_number - 1)`.
                 The first cycle is cycle 1.
    """
    @typechecked
    def __init__(self,
                 lr0: FloatTensorLike,
                 lr1: FloatTensorLike,
                 gamma: FloatTensorLike,
                 half_cycle_length: FloatTensorLike,
                 cycle_profile: str = "linear",
                 name: str = "ExponentialCyclicalLearningRate"):
        super().__init__(lr0=lr0,
                         lr1=lr1,
                         half_cycle_length=half_cycle_length,
                         cycle_scale=lambda cycle_number: self.gamma**(cycle_number - 1),
                         cycle_profile=cycle_profile,
                         name=name)
        self.gamma = gamma

    def get_config(self):
        return {"lr0": self.lr0,
                "lr1": self.lr1,
                "gamma": self.gamma,
                "half_cycle_length": self.half_cycle_length,
                "cycle_profile": self.cycle_profile,
                "name": self.name}
