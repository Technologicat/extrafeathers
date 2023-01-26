"""Visualize the architecture of the model, and an example of each of the resnet block types.

The outputs are aesthetically not very pleasing, and a raster format (such as png) is not ideal
for use in publications, but the idea is for now to produce *something* automatically.

(Despite the name, "visarch" is not a Warhammer 40k unit.)
"""

import os
import sys

import tensorflow as tf
import visualkeras

from .config import output_dir
from . import main
from . import resnet
from . import util

# Optionally take a directory name from the command line
# (so that we can visualize architectures of completed models later
#  after the directory has been renamed for archival purposes).
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = output_dir
if data_dir[-1] != os.path.sep:
    data_dir += os.path.sep

# --------------------------------------------------------------------------------

model_dir = f"{data_dir}model/final"
main.model.my_load(model_dir)

# --------------------------------------------------------------------------------

tf.keras.utils.plot_model(main.model.encoder,
                          to_file=f"{data_dir}arch_encoder_graph.png",
                          show_shapes=True)
tf.keras.utils.plot_model(main.model.decoder,
                          to_file=f"{data_dir}arch_decoder_graph.png",
                          show_shapes=True)
visualkeras.layered_view(main.model.encoder).save(f"{data_dir}arch_encoder_vis.png")
visualkeras.layered_view(main.model.decoder).save(f"{data_dir}arch_decoder_vis.png")

# --------------------------------------------------------------------------------

# The details here depend on the exact architecture set up in `main.py`.
# The values here correspond to "ResNet attempt 7".
# The important point is to give a general idea of the structure of these blocks.
#
# Encoder ResNet blocks (first instance of each in the encoder structure)
b = resnet.ConvolutionBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                              bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(28, 28, 1))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_convolutionblock2d.png",
                          show_shapes=True)

b = resnet.IdentityBlock2D(filters=32, kernel_size=3, activation=tf.keras.layers.PReLU,
                           bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(14, 14, 32))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_identityblock2d.png",
                          show_shapes=True)

b = resnet.ProjectionBlock2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                             bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(14, 14, 32))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_projectionblock2d.png",
                          show_shapes=True)

# Decoder ResNet blocks (first instance of each in the decoder structure)
b = resnet.IdentityBlockTranspose2D(filters=256, kernel_size=3, activation=tf.keras.layers.PReLU,
                                    bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(7, 7, 256))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_identityblocktranspose2d.png",
                          show_shapes=True)

b = resnet.ProjectionBlockTranspose2D(filters=128, kernel_size=3, activation=tf.keras.layers.PReLU,
                                      bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(7, 7, 256))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_projectionblocktranspose2d.png",
                          show_shapes=True)

b = resnet.ConvolutionBlockTranspose2D(filters=64, kernel_size=3, activation=tf.keras.layers.PReLU,
                                       bottleneck_factor=2)
m = util.layer_to_model(b, input_shape=(7, 7, 128))
tf.keras.utils.plot_model(m,
                          to_file=f"{data_dir}arch_convolutionblocktranspose2d.png",
                          show_shapes=True)
