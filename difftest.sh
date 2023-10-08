#!/bin/bash
# The async allocator helps with VRAM fragmentation, so it is good when pushing the limits.
TF_GPU_ALLOCATOR=cuda_malloc_async python -m demo.vae.difftest
