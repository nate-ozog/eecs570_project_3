#ifndef XS_CUH
#define XS_CUH

#include "nw_general.h"
#include "xs_core.cuh"
#include "cuda_error_check.cuh"

std::mutex mtx;

std::pair<uint8_t *, int> xs_man(
  const char * t,
  const char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
  num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(uint8_t);
  num_GPU_mem_bytes += tlen * sizeof(char);
  num_GPU_mem_bytes += qlen * sizeof(char);

  // Malloc memory for our program in a critical section.
  mtx.lock();
  void * GPU_mem = NULL;
  size_t free = 0, total = 0;
  // REMOVE.....v...this is overprovision to allow normal GPU
  // operation on a system that is not used purely for compute!
  while (free < 2 * num_GPU_mem_bytes)
    cuda_error_check(cudaMemGetInfo(&free, &total));
  cuda_error_check(cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes));
  mtx.unlock();

  // Run our kernel manager.
  std::pair<uint8_t *, int> res
    = xs_t_geq_q_man(t, q, tlen, qlen, mis_or_ind, GPU_mem);
  cuda_error_check(cudaFree(GPU_mem));
  return res;
}

#endif