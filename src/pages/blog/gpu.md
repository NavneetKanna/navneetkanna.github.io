---
layout: ../../layouts/BlogPost.astro
title: "GPU Programming"
date: 2025-04-02
---

I'm documenting my GPU programming journey through this blog post. I do not have a nvidia gpu, but I do however have apple m2. The concepts are similar irresepective of the manufacturer.

### Grid, Threadblocks, Warps

- Given a grid, which can be 1D, 2D or 3D, for example, a matrix or an image, it is subdivided into (thread)blocks of size n, which is further subdivided into groups of 32. 
- We can think of a 2D matrix, lets say 4096x4096, as a grid of 4096x4096 threads. This grid is divided into blocks of size n, lets say 1024 (32 width-wise, 32-height wise, 1 depth-wise).
The 32x32x1 blocks are further divided into 1D group of size 32 called warps. So in this case, there are 32 warps for this block.
- The number 1024 and 32 are not choosen at random, altough it varies from architecture to architure, it is genereally 1024 and 32.
- All threads in a warp have consecutive thread ID value.

### Streaming Multiprocessor (SM)

- These links should give sufficient information to start with
    - https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor
    - https://stevengong.co/notes/Streaming-Multiprocessor
    - https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
    - Programming Massively Parallel Processors: A Hands-on Approach book
- Basically, it is the heart of the GPU that consists of cores (CUDA Cores / Streaming Processor (SP)) which execute the instructions in parallel.
- Every thread in a block is guarenteed to run on the same SM.
- For example, on the A100 GPU, there are 108 SM's with 64 cores each, totalling to 6912 cores on the entire GPU. The SM is organized into 4 processing blocks. So, at any given time 4 warps are running simultaneously in 1 SM.
 
### Single Instruction Multiple Data (SIMT) Architecture

- A SM follows SIMT model, which means at any instant in time, one instruction is fetched and executed by all threads in the warp.
- These threads apply the same instruction to different portions of the data.
