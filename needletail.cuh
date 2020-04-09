#ifndef NEEDLETAIL_CUH
#define NEEDLETAIL_CUH

#include "needletail_general.hpp"
#include "needletail_core.cuh"
#include "cuda_error_check.cuh"

std::pair<uint8_t *, int> nt_man (
  char * t,
  char * q,
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
  cuda_error_check( cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes) );
  // Create a stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::pair<uint8_t *, int> nt_res
    = nt_man(t, q, tlen, qlen, mis_or_ind, GPU_mem, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(GPU_mem);
  return nt_res;
}

#endif