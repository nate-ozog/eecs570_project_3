#ifndef XS_CUH
#define XS_CUH

#include "nw_general.hpp"
#include "xs_core.cuh"
#include "cuda_error_check.cuh"

uint32_t * xs_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {

  uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
  num_GPU_mem_bytes += ceil((tlen+1) / float(PTRS_PER_ELT)) * (qlen+1) * sizeof(uint32_t);
  num_GPU_mem_bytes += tlen * sizeof(char);
  num_GPU_mem_bytes += qlen * sizeof(char);
  // Malloc memory for our program.
  void * GPU_mem = NULL;
  cuda_error_check( cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes) );
  // Initialize allocated memory to zeros.
  cuda_error_check( cudaMemset(GPU_mem, 0, num_GPU_mem_bytes) );
  // Create a stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  uint32_t * mat = xs_t_geq_q_man(t, q, tlen, qlen, mis_or_ind, GPU_mem, &stream);
  cudaStreamSynchronize(stream);
  cudaFree(GPU_mem);

  /* // TEMP: UNCOMMENT FOR MATRIX PRINTING! */
  /* for (int i = 0; i <= qlen; ++i) { */
  /*   for (int j = 0; j <= tlen; ++j) */
  /*     std::cout << std::setfill(' ') << std::setw(5) */
  /*       << mat[(tlen+1) * i + j] << " "; */
  /*   std::cout << std::endl; */
  /* } */

  return mat;
}

#endif
