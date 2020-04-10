#ifndef XS_CUH
#define XS_CUH

#include <memory>

#include <mutex>
#include <list>
#include <utility>
// #include <unordered_map>

#include "nw_general.h"
#include "xs_core.cuh"
#include "cuda_error_check.cuh"

uint8_t * xs_man(
  const char * t,
  const char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
  num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(int);
  num_GPU_mem_bytes += tlen * sizeof(char);
  num_GPU_mem_bytes += qlen * sizeof(char);
  // Malloc memory for our program.
  void * GPU_mem = NULL;
  while(cudaErrorMemoryAllocation == cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes));
  // Create a stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  uint8_t * mat = xs_t_geq_q_man(t, q, tlen, qlen, mis_or_ind, GPU_mem, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(GPU_mem);
  return mat;
}

#endif