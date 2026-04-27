---
layout: post
title: Histogram in CUDA 
date: 2026-04-23
description: Exploring different optimization methods in the implementation of Histograms in CUDA. 
tags: [CUDA, GPU, Parallel Programming, notes, histogram]
categories: []
giscus_comments: true
---
This post actually isn't a blog post but it's a compilation of my lecture notes on parallel programming using CUDA. So it's not a tutorial per se and *it requires prior knowledge on CUDA and Nvidia GPU architecture*. 

A traditional histogram implementation in C runs in a sequential way, where using a for loop we iterate through the input space. CUDA allows us to exploit its thread parallelism to execute the same task without the need to iterate through the input space.

In this implementation we will use a Histogram with 6 chars per bin, so 5 bins for the 26 letters of the alphabet. 

```c 
#define MAX_LENGTH 50000000

#define WARPSIZE 32
#define BLOCKDIM WARPSIZE
#define GRIDDIM  2048

#define CHAR_PER_BIN  6
#define ALPHABET_SIZE 26
#define BIN_NUM       ((ALPHABET_SIZE - 1) / CHAR_PER_BIN + 1)
#define FIRST_CHAR    'a'


void sequential_histogram(const char *data, unsigned int *histogram, const int length) {
  for (int i = 0; i < length; i++) {
    int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) // check if we have an alphabet char
      histogram[alphabet_position / CHAR_PER_BIN]++;                 // we group the letters into blocks of 6
  }
}
```

When designing CUDA kernels the design can be modeled around the input data or the output data. The most common approach is the input oriented design which you can see in the ```add.cu``` examples. 

In this case our task is to parallelize the sequential histogram. The same way as in CPU, also in CUDA the memory accesses have various levels of memory hierarchy each with its own set of advantages and disadvantages. The two costs we'll be chipping away at throughout this post are: how fast can we *read* the input (coalescing), and how much do threads contend on the *writes* to the histogram (atomics).


First Naive implementation: 
```c

__global__ void histogram_kernel(const char *__restrict__ data, unsigned int *__restrict__ histogram, const int length) {
  const unsigned int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  const int section_size    = (length - 1) / (blockDim.x * gridDim.x) + 1;
  const int start           = tid * section_size;
  // All threads handle blockDim.x * gridDim.x
  // consecutive elements
  for (size_t k = 0; k < section_size; k++) {
    if (start + k < length) {
      const int alphabet_position = data[start + k] - FIRST_CHAR;
      if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
        atomicAdd(&(histogram[alphabet_position / CHAR_PER_BIN]), 1);
    }
  }
}
``` 
In this implementation each of our threads iterate through a section of the input text sized `section_size = (length - 1) / (blockDim.x * gridDim.x) + 1;` and then each char after having its index computed gets committed to the `histogram` on the global memory. 

One approach we can take is the usage of interleaved memory access or interleaved coarsening. 


Basically, in this approach each thread also accesses multiple input elements, but since memory access at the warp level happens in lockstep (all 32 threads issue their loads at the same cycle), we can exploit the interleaving so that the addresses produced by the warp are contiguous. The hardware's coalescing logic can then fold them into a single memory transaction for the whole warp.

So in this case for example, with `stride = blockDim.x * gridDim.x`, thread 0 reads memory addresses [0, stride, 2·stride, ...] whereas thread 1 reads [1, 1+stride, 1+2·stride, ...] and so on. At any given iteration the 32 threads of a warp read 32 *adjacent* bytes, which the hardware coalesces into one transaction. 

```c
__global__ void
    histogram_kernel(const char *__restrict__ data, unsigned int *__restrict__ histogram, const int length) {
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // All threads handle blockDim.x * gridDim.x consecutive elements in each iteration
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histogram[alphabet_position / CHAR_PER_BIN]), 1);
  }
}

```

With this kernel we have improved upon our read speeds but not our write speeds. 

One aspect we should consider is the costs of the penalties caused by the atomicAdd instructions coming from thousands of threads. One way to minimize these penalties is to implement a privatization of memory, and the real win comes from putting the private histogram in *shared memory*, which has roughly 100× lower latency than global memory for atomic operations.

In this case we will introduce a block level shared histogram called `histo_s`. In this particular kernel the generation of the histogram is going to happen in two phases. In the first phase the threads are going to generate a block level histogram on the shared memory, whose access speed is far superior to the global memory. Secondly the shared memory histogram is going to be committed to the global memory through a strided loop, where each thread handles a subset of the bins. This way the commit is parallelized across the block, and only non-zero bins trigger a global atomicAdd, so we end up with at most `BIN_NUM` global atomics per block at the very end instead of one per input sample.

``` c 

__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { histo_s[binIdx] = 0u; }
  __syncthreads();
  // Histogram
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histo_s[alphabet_position / CHAR_PER_BIN]), 1);
  }
  __syncthreads();
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}

```

We can improve the performance of the kernel by further exploiting CUDA's memory hierarchies. Shared memory is fast, but do you know what's faster... registers!

So in this implementation we keep the interleaved access, but our threads will first commit their findings on their own private histogram held in registers (private meaning visible only to the thread itself), then these per-thread histograms will be folded into the shared memory histogram, which will at the end commit the final histogram to the global memory. IK it sounds more complicated, but in case you have a limited amount of bins it scales fairly well.

FYI: in CUDA, a small fixed-size array with compile time known indices (like `histo_p[BIN_NUM]` here, with `BIN_NUM = 5`) can be promoted to registers by the compiler. If `BIN_NUM` grows too large, or the indexing becomes dynamic, the array spills into what CUDA calls "local memory", which is actually stored in global DRAM and is one of the *slowest* memory spaces. So this trick only works when you have very few bins and the loop indices are simple enough for the compiler to unroll. If either condition breaks, this kernel gets dramatically worse than the previous one.

```c 

__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  unsigned int histo_p[BIN_NUM];
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int i = 0; i < BIN_NUM; i++) histo_p[i] = 0u;
  // Histogram
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      histo_p[alphabet_position / CHAR_PER_BIN] += 1;
  }
  // Commit to shared memory
#pragma unroll
  for (unsigned int binIdx = 0; binIdx < BIN_NUM; binIdx++) {
    const unsigned int binValue = histo_p[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histo_s[binIdx]), binValue);
    }
  }
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}

```

The approach above is honestly great when bins are few, but it has a weakness: if you have many bins or the input itself has clusters of repeated characters, you're still doing one atomic per sample. So here's a different angle on the same problem, exploiting locality in the input stream rather than adding more memory tiers.

The idea is to keep two scalars per thread, a `prevBinIdx` and an `accumulator`, and stream through the input. As long as consecutive samples fall in the same bin, we just bump the accumulator locally with no atomic at all. Only when the bin changes do we emit a single atomicAdd to the shared histogram for the whole run. After having iterated through the whole sequence we commit the shared histogram to the global memory the same way as before. 

This is especially nice for inputs with locality. In the best case, a run of N consecutive same-bin samples becomes one atomic instead of N. 

```c 

__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { histo_s[binIdx] = 0u; }
  __syncthreads();
  // Histogram
  unsigned int accumulator = 0;
  int prevBinIdx           = -1;
  for (unsigned int i = tid; i < length; i += stride) {
    int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
      const int bin = alphabet_position / CHAR_PER_BIN;
      if (bin == prevBinIdx) {
        ++accumulator;
      } else {
        if (accumulator > 0) {
          atomicAdd(&(histo_s[prevBinIdx]), accumulator);
        }
        accumulator = 1;
        prevBinIdx  = bin;
      }
    }
  }
  if (accumulator > 0) {
    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
  }
  __syncthreads();
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}

```

## Summary

| Kernel | Read pattern | Atomic location | Contention scope | Requires |
|---|---|---|---|---|
| Naive | uncoalesced | global | grid-wide | — |
| Interleaved | coalesced | global | grid-wide | — |
| Shared privatization | coalesced | shared | block-wide | `BIN_NUM` shared mem |
| Register + shared | coalesced | shared | block-wide | `BIN_NUM` registers per thread |
| Accumulator | coalesced | shared (reduced) | block-wide | locality in input |

Picking between the last two is a bit situational. The register version wins when `BIN_NUM` is small enough to fit in registers and the input is essentially random (no clustering to exploit). The accumulator version wins when the input has any locality, regardless of `BIN_NUM`, since it works directly on top of the shared-memory kernel without needing the bin count to fit anywhere special. For natural language text with 5 bins, the accumulator approach is basically in its best case scenario.

