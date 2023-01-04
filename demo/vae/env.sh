#!/bin/bash
#
# from pip:
#   tensorflow
#   tensorflow-probability
#   tensorboard
#   keras
# OR
#   tf-nightly
#   tensorflow-probability
#   tb-nightly
#   keras-nightly
#
# tensorflow-gpu is old (deprecated?, from 1.x days), don't install it.
# As of 2.x, the base `tensorflow` package supports GPU.
#
# Then install some pip packages from NVIDIA:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
#   nvidia-tensorrt  (should pull in cuda_runtime, cuda_nvrtc, cublas, cudnn)
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
export LD_LIBDARY_PATH=/usr/lib/x86_64-linux-gnu/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cublas/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cufft/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/curand/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusolver/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusparse/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/tensorrt/
