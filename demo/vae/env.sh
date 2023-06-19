#!/bin/bash
#
# Set up environment variables for GPU acceleration (CUDA) of the VAE model.
#
# Requirements:
#
# from pip:
#   tensorflow
#   tensorflow-probability
#   tensorboard
# OR to get the bleeding edge,
#   tf-nightly
#   tensorflow-probability
#   tb-nightly
#
# Notes:
#   - `tensorflow-gpu` is old (deprecated?, from 1.x days), don't install it.
#     As of 2.x, the base `tensorflow` package supports GPU.
#   - The separate `keras` package is old; as of TensorFlow 2.x,
#     Keras has moved into `tf.keras` and is now distributed as part of TensorFlow.
#   - When reading examples online, be aware that as of January 2023, the Keras API
#     has changed slightly from how it was in the final separate `keras` package;
#     for example, `plot_model` is now directly in `keras.utils`, and there is no
#     separate `plot_utils` submodule.
#   - See also `visualkeras` to easily visualize simple encoder/decoder designs
#     (when you have a `tf.keras.Model` instance and want to get a diagram).
#
# Then install some pip packages from NVIDIA:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
#   nvidia-tensorrt  (should pull in cuda_runtime, cuda_nvrtc, cublas, cudnn)
#      nvidia_cuda_nvrtc_cu11  (in case it doesn't, this is the package name)
#      nvidia_cuda_runtime_cu11
#      nvidia_cublas_cu11
#      nvidia_cudnn_cu11
#   nvidia_cufft_cu11  (with appropriate cuda version; check ~/.local/lib/python3.10/site-packages/nvidia*)
#   nvidia_curand_cu11
#   nvidia_cusolver_cu11
#   nvidia_cusparse_cu11
#   nvidia_cuda_nvcc_cu11
#
# May need to install specific versions, e.g.
#   pip install nvidia_cudnn_cu11==8.6.0.163
# (see error messages, if any, produced when running TensorFlow; should say if there is a version mismatch)

# ptxas
export PATH=$PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin
# cuda directory containing nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/jje/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc

# libcuda.so.1 (for tf-nightly 2.12)
# This is the system libcuda from libnvidia-compute-xxx, where xxx is the version number (e.g. 525).
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cublas/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cufft/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/curand/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusolver/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusparse/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/tensorrt/
