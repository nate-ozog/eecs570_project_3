#ifndef NEEDLETAIL_THREADS_CUH
#define NEEDLETAIL_THREADS_CUH

#include "needletail_general.hpp"
#include "cuda_error_check.cuh"
#include "needletail_kernels.cuh"
#include "pools.hpp"

std::tuple<char *, char *, int> needletail_stream_single (
  const char * t,
  const char * q,
  uint32_t     tlen,
  uint32_t     qlen,
  signed char  mis_or_ind,
  bool         swap_t_q
) {
  std::tuple<char *, char *, int> results;
  std::pair<char *, char *> algn;
  cudaStream_t stream;
  uint64_t     device_mem_bytes;
  uint64_t     host_mem_bytes;
  void *       device_mem_ptr = NULL;
  uint8_t *    host_mem_ptr = NULL;
  uint32_t     max_strides;
  uint32_t     shared_mem_size;
  int *        col = NULL;
  int *        col_d = NULL;
  uint8_t *    mat_d = NULL;
  char *       t_d = NULL;
  char *       q_d = NULL;
  int *        opt_score;

  cudaStreamCreate( &stream );

  // Precompute needed values.
  max_strides = 2;
  shared_mem_size = (BLOCK_SIZE + 1) * sizeof(int) * max_strides * 3;
  col = new int [tlen + qlen + 1];
  for (size_t i = 0; i < tlen + qlen + 1; ++i)
    col[i] = i * mis_or_ind;

  // Malloc memory on device and host.
  device_mem_bytes  = (qlen + tlen + 1) * sizeof(int);
  device_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(uint8_t);
  device_mem_bytes += tlen * sizeof(char);
  device_mem_bytes += qlen * sizeof(char);
  host_mem_bytes    = (tlen + 1) * (qlen + 1) * sizeof(uint8_t);

  // Allocate memory
  device_mem_ptr = device_pool.malloc( device_mem_bytes );
  host_mem_ptr = (uint8_t *) host_pool.malloc( host_mem_bytes );
  opt_score = (int *) host_pool.malloc( sizeof(int) );

  // Compute argument addresses
  col_d = (int *) device_mem_ptr;
  mat_d = (uint8_t *) (col_d + (tlen + qlen + 1));
  t_d   = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  q_d   = t_d + tlen;

  // Schedule host to device memory copies.
  cudaMemcpyAsync(col_d, col, (tlen + qlen + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, stream);

  // Schedule kernel launches.
  needletail_kernel <<< 1, BLOCK_SIZE, shared_mem_size, stream >>>
    (t_d, q_d, tlen, qlen, col_d, max_strides, mis_or_ind, mat_d);

  // Schedule device to host memory copy.
  cudaMemcpyAsync(host_mem_ptr, mat_d, host_mem_bytes, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(opt_score, col_d + (qlen + tlen), sizeof(int), cudaMemcpyDeviceToHost, stream);

  // Synchronize with stream and start cleanup.
  cudaStreamSynchronize(stream);

  // Free device memory.
  device_pool.free(device_mem_ptr);

  cudaStreamDestroy(stream);

  // Backtrack using the matrix in host memory
  algn = nw_ptr_backtrack(host_mem_ptr, swap_t_q, t, q, tlen, qlen);

  // Return the aligned strings and optimal score.
  std::get<0>(results) = algn.first;
  std::get<1>(results) = algn.second;
  std::get<2>(results) = *opt_score;

  // Free the host memory
  host_pool.free( host_mem_ptr );
  host_pool.free( opt_score );

  delete [] col;
  return results;
}

#endif
