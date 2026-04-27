---
layout: post
title: Convolutions in CUDA 
date: 2026-04-27
description: Exploring different optimization methods in the implementation of Convolutions in CUDA. 
tags: [CUDA, GPU, Parallel Programming, notes, convolution]
categories: []
giscus_comments: true
---

As indicated also in my previous CUDA post, this post is also a prettified version of my lecture notes. So it's not a proper tutorial and it requires prior knowledge.

Mathematically speaking, convolution is an operator on two functions (or matrices) that produces a third one, where the output is the input modified by the other to extract or emphasize certain features. [Definition adapted from this source.](https://towardsdatascience.com/computer-vision-convolution-basics-2d0ae3b79346/)

![convolution gif](https://media.geeksforgeeks.org/wp-content/uploads/20230216175224/how-to-apply-a-2d-convolution-operation-in-pytorch.gif)

So basically at the end of the day a convolution in our use case is a bunch of element-wise multiplications followed by a sum, repeated for every output pixel.

Let's define our CPU based approach.

```c
void convolution_cpu(input_type *input,
                     const input_type *filter,
                     input_type *output,
                     const int width,
                     const int height,
                     const int filter_size,
                     const int filter_radius) {
  for (int outRow = 0; outRow < width; outRow++) {
    for (int outCol = 0; outCol < height; outCol++) {
      input_type value{0.0f};
      for (int row = 0; row < filter_size; row++)
        for (int col = 0; col < filter_size; col++) {
          int inRow = outRow - filter_radius + row;
          int inCol = outCol - filter_radius + col;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            value += filter[row * filter_size + col] * input[inRow * width + inCol];
          }
        }
      output[outRow * width + outCol] = value;
    }
  }
}
```

The two costs we'll be chipping away at throughout this post are: how much *global memory traffic* we generate per output pixel, and how well we *exploit data reuse* between neighboring output pixels (since adjacent outputs read overlapping input regions).

## Naive implementation using CUDA

In this approach every thread is assigned to calculate one output pixel. The thread reads its `FILTER_SIZE × FILTER_SIZE` neighborhood from the input, multiplies it with the filter, and writes the sum to the output.

```c
__global__ void convolution_basic_kernel(const input_type *__restrict__ input,
                                         const filter_type *__restrict__ filter,
                                         input_type *__restrict__ output,
                                         const int width,
                                         const int height,
                                         const int filter_size,
                                         const int filter_radius) {
  const int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  input_type value{0.0f};
  for (int row = 0; row < filter_size; row++)
    for (int col = 0; col < filter_size; col++) {
      const int inRow = outRow - filter_radius + row;
      const int inCol = outCol - filter_radius + col;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        value += filter[row * filter_size + col] * input[inRow * width + inCol];
      }
    }
  output[outRow * width + outCol] = value;
}
```

By exploiting thread parallelism on CUDA we managed to reduce the number of nested loops from 4 to 2 — every output pixel is computed in parallel.

This already works much better than the CPU approach, however it has an extremely low arithmetic intensity of roughly *0.25 OP/B*: for every output pixel, each thread reads two 4-byte values from global memory (one input element and one filter element) and performs two operations (one multiply, one add). That's 2 ops per 8 bytes, so 0.25 OP/B. This puts us deep in the memory-bound region, thus we're not exploiting the GPU's compute throughput at all.

## Constant memory for the filter

One trick we can pull is using a different memory space to increase our arithmetic intensity. Specifically, we can put the filter in *constant memory*.

Constant memory is a region that cannot be modified by threads during kernel execution. It's small (64 KB). When all threads in a warp read the same address (which is exactly what happens here, since every thread in the warp uses the same filter coefficient at any given step), the read is served essentially for free from the constant cache. So we can effectively eliminate the filter from our DRAM bandwidth, halving the bytes read per operation. Arithmetic intensity bumps up to *0.5 OP/B*.

```c
__global__ void convolution_constant_mem_kernel(const input_type *__restrict__ input,
                                                input_type *__restrict__ output,
                                                const int width,
                                                const int height) {
  const int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  input_type value{0.0f};
#pragma unroll
  for (int row = 0; row < FILTER_SIZE; row++)
#pragma unroll
    for (int col = 0; col < FILTER_SIZE; col++) {
      const int inRow = outRow - FILTER_RADIUS + row;
      const int inCol = outCol - FILTER_RADIUS + col;
      if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        value += constant_filter[row][col] * input[inRow * width + inCol];
      }
    }
  output[outRow * width + outCol] = value;
}
```

The filter is declared at file scope using the `__constant__` keyword:

```c
__constant__ filter_type constant_filter[FILTER_SIZE][FILTER_SIZE];
```

Better, but we're still hammering global memory for the input. The bigger win comes from noticing that adjacent output pixels read overlapping input regions so if we cooperate within a block, we can read each input pixel from DRAM exactly once and reuse it many times.

## Tiled convolution with full halo

The classic tiling approach: each block loads an input tile into shared memory once, then every thread in the block computes its output by reading exclusively from shared memory.

The catch is that each output pixel needs a `FILTER_SIZE × FILTER_SIZE` neighborhood, so to compute a `OUT_TILE_DIM × OUT_TILE_DIM` block of outputs we need to load a slightly larger region of `IN_TILE_DIM × IN_TILE_DIM = (OUT_TILE_DIM + 2·FILTER_RADIUS)²` input pixels — that's the actual tile plus a *halo* of `FILTER_RADIUS` extra pixels on each side. The block is launched with `IN_TILE_DIM × IN_TILE_DIM` threads so that every thread loads exactly one input pixel; the threads in the halo region load but don't compute any output.

```c
__global__ void convolution_tiled_kernel(const input_type *__restrict__ input,
                                         input_type *__restrict__ output,
                                         const int width,
                                         const int height) {
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int inCol = bidx * OUT_TILE_DIM + tidx - FILTER_RADIUS;
  const int inRow = bidy * OUT_TILE_DIM + tidy - FILTER_RADIUS;

  // Load input tile
  __shared__ input_type input_shared[IN_TILE_DIM][IN_TILE_DIM];
  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
    input_shared[tidy][tidx] = input[inRow * width + inCol];
  } else {
    input_shared[tidy][tidx] = 0.0;
  }
  __syncthreads();

  // Compute output elements
  const int tileCol = tidx - FILTER_RADIUS;
  const int tileRow = tidy - FILTER_RADIUS;
  if (inCol >= 0 && inCol < width && inRow >= 0 && inRow < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
      input_type output_value{0.0f};
#pragma unroll
      for (int row = 0; row < FILTER_SIZE; row++)
#pragma unroll
        for (int col = 0; col < FILTER_SIZE; col++)
          output_value += constant_filter[row][col] * input_shared[tileRow + row][tileCol + col];
      output[inRow * width + inCol] = output_value;
    }
  }
}
```

Tiling reduces DRAM accesses because each input pixel is read from global memory exactly once per block instead of once per output pixel that needs it. The price is shared memory usage and a `__syncthreads()` after the load. There's also some thread inefficiency: the halo threads load data but never produce an output, so a fraction of the threads sit idle during the compute phase.

## Cached tiled convolution

Here's a different angle on the tiling problem: what if we let the *hardware cache* handle the halo for us?

Notice that the halo pixels around block A's tile are exactly the inner-tile pixels of the neighboring blocks. When those neighboring blocks run on the same SM, their threads read those same pixels from global memory, which brings them into the L1/L2 cache. So when a thread in block A needs a halo pixel, going back to global memory is almost certainly an L2 cache hit.

That's the idea behind this version: load only the inner `TILE_DIM × TILE_DIM` tile into shared memory (no halo), and when a thread needs a halo pixel during the compute loop, fetch it directly from global memory and **trust** the cache to make it cheap. The block can now be launched with `TILE_DIM × TILE_DIM` threads — every thread loads one pixel and every thread computes one output, no idle halo threads.

```c
__global__ void convolution_tiled_kernel(const input_type *__restrict__ input,
                                         input_type *__restrict__ output,
                                         const int width,
                                         const int height) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int col = blockIdx.x * TILE_DIM + tidx;
  const int row = blockIdx.y * TILE_DIM + tidy;

  // Load input tile (no halo)
  __shared__ input_type input_shared[TILE_DIM][TILE_DIM];
  if (row < height && col < width) {
    input_shared[tidy][tidx] = input[row * width + col];
  } else {
    input_shared[tidy][tidx] = 0.0;
  }
  __syncthreads();

  // Compute output elements
  if (col < width && row < height) {
    float PValue = 0.0f;
#pragma unroll
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
#pragma unroll
      for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
        if (tidx - FILTER_RADIUS + fCol >= 0 && tidx - FILTER_RADIUS + fCol < TILE_DIM &&
            tidy - FILTER_RADIUS + fRow >= 0 && tidy - FILTER_RADIUS + fRow < TILE_DIM) {
          PValue += constant_filter[fRow][fCol] *
                    input_shared[tidy - FILTER_RADIUS + fRow][tidx - FILTER_RADIUS + fCol];
        } else {
          if (row - FILTER_RADIUS + fRow >= 0 && row - FILTER_RADIUS + fRow < height &&
              col - FILTER_RADIUS + fCol >= 0 && col - FILTER_RADIUS + fCol < width) {
            PValue += constant_filter[fRow][fCol] *
                      input[(row - FILTER_RADIUS + fRow) * width + col - FILTER_RADIUS + fCol];
          }
        }
      }
    }
    output[row * width + col] = PValue;
  }
}
```

The "cached" name reflects the fact that we're not manually staging the halo into shared memory, we're trusting the hardware L1/L2 cache to keep the neighboring tiles pixels around. The shared memory holds only what's exclusively ours; the halo is "cached for free" by the hardware.

Tradeoffs versus the full-halo version: the shared memory footprint is smaller (`TILE_DIM²` vs `(TILE_DIM + 2·FILTER_RADIUS)²`), which helps maximise block occupancy since shared memory limits how many blocks can run per SM. There are no idle halo threads, and the loading code is simpler. The downside is the boundary check inside the inner loop, which can cause branch divergence, which is bad in a SIMT architecture, and the performance depends on the cache actually doing its job.

## Summary

| Kernel | Filter location | Input location | Halo handling | Notes |
|---|---|---|---|---|
| Naive | global | global | per-thread reads | ~0.25 OP/B, memory-bound |
| Constant filter | constant cache | global | per-thread reads | ~0.5 OP/B, broadcast reads for filter |
| Tiled (full halo) | constant cache | shared | loaded into shared | no global reads during compute, idle halo threads |
| Tiled (cached) | constant cache | shared | global (L1/L2 hit) | smaller shared footprint, branch divergence in inner loop |