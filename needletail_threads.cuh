#ifndef NEEDLETAIL_THREADS_CUH
#define NEEDLETAIL_THREADS_CUH

#include "nw_general.h"
#include "cuda_error_check.cuh"
#include "needletail_kernels.cuh"
#include "pools.h"

std::mutex mtx;

struct stream_args {
  void * mem;
  uint8_t * mat;
  int * row0_d;
  int * row1_d;
  int * row2_d;
  uint8_t * mat_d;
  char * t_d;
  char * q_d;
};

std::pair<char *, char *> needletail_stream_single (
  const char *t,
  const char *q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  bool swap_t_q
) {
  std::pair<char *, char *> algn;
  cudaStream_t stream;
  uint64_t device_mem_bytes;
  uint64_t host_mem_bytes;
  void     *device_mem_ptr = NULL;
  uint8_t  *host_mem_ptr   = NULL;
  int      *row0_d;
  int      *row1_d;
  int      *row2_d;
  uint8_t  *mat_d;
  char     *t_d;
  char     *q_d;

  cudaStreamCreate( &stream );

  // Compute allocation sizes
  device_mem_bytes  = 3 * (tlen + 1) * sizeof(int);
  device_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(int);
  device_mem_bytes += tlen * sizeof(char);
  device_mem_bytes += qlen * sizeof(char);
  host_mem_bytes    = (tlen + 1) * (qlen + 1) * sizeof(uint8_t);

  // Allocate memory
  pthread_mutex_lock( &device_pool_lock );
  device_mem_ptr = device_pool.malloc( device_mem_bytes );
  //device_pool.print_pool();
  pthread_mutex_unlock( &device_pool_lock );

  pthread_mutex_lock( &host_pool_lock );
  host_mem_ptr = (uint8_t*) host_pool.malloc( host_mem_bytes );
  pthread_mutex_unlock( &host_pool_lock );

  //cudaMalloc( &device_mem_ptr, device_mem_bytes );
  //cudaHostAlloc( &host_mem_ptr, host_mem_bytes, cudaHostAllocDefault );

  // Compute argument addresses
  row0_d = (int *) device_mem_ptr;
  row1_d = row0_d + (tlen + 1);
  row2_d = row1_d + (tlen + 1);
  mat_d  = (uint8_t *) (row2_d + (tlen + 1));
  t_d    = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  q_d    = t_d + tlen;

  // Schedule host to device memory copies
  cudaMemcpyAsync( t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, stream );
  cudaMemcpyAsync( q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, stream );

  // Schedule kernel launches
  needletail_init_kernel <<< divide_then_round_up((tlen + 1), BLOCK_SIZE), BLOCK_SIZE, 0,  stream >>>
      (tlen, qlen, mis_or_ind, row0_d, row1_d, mat_d);
  needletail_comp_kernel <<< 1, BLOCK_SIZE, 0,  stream >>>
    (t_d, q_d, tlen, qlen, mis_or_ind, row0_d, row1_d, row2_d, mat_d);

  // Schedule device to host memory copy
  cudaMemcpyAsync( host_mem_ptr, mat_d, (tlen + 1) * (qlen + 1) * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream );

  // Synchronize with stream and start cleanup
  cudaStreamSynchronize( stream );

  pthread_mutex_lock( &device_pool_lock );
  device_pool.free( device_mem_ptr );
  pthread_mutex_unlock( &device_pool_lock );

  //cudaFree( device_mem_ptr );
  cudaStreamDestroy( stream );

  // Backtrack using the matrix in host memory
  algn = nw_ptr_backtrack( host_mem_ptr, swap_t_q, t, q, tlen, qlen );

  // Free the host memory

  pthread_mutex_lock( &host_pool_lock );
  host_pool.free( host_mem_ptr );
  pthread_mutex_unlock( &host_pool_lock );

  //cudaFreeHost( host_mem_ptr );

  // Return the aligned strings
  return algn;
}

#endif