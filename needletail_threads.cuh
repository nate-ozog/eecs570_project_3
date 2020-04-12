#ifndef NEEDLETAIL_THREADS_CUH
#define NEEDLETAIL_THREADS_CUH

#include "nw_general.h"
#include "testbatch.hpp"
#include "cuda_error_check.cuh"
#include "needletail_kernels.cuh"

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

void needletail_stream_batch (
  uint32_t batch_size,
  bool * swap_t_q,
  Test_t * tests
) {

  // Create an array of stream arguments and streams.
  stream_args s_args[batch_size];
  cudaStream_t s[batch_size];

  // Grab the scheduling lock.
  mtx.lock();

  // Create the streams, claim GPU memory, and setup arguments.
  for (size_t i = 0; i < batch_size; ++i) {
    cudaStreamCreate(&s[i]);
    uint32_t tlen = tests[i].s1_len;
    uint32_t qlen = tests[i].s2_len;
    uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
    num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(uint8_t);
    num_GPU_mem_bytes += tlen * sizeof(char);
    num_GPU_mem_bytes += qlen * sizeof(char);
    size_t free = 0, total = 0;
    while (free < 2 * num_GPU_mem_bytes)
      cudaMemGetInfo(&free, &total);
    cudaMalloc((void **) & s_args[i].mem, num_GPU_mem_bytes);
    cudaHostAlloc((void**)& s_args[i].mat, (tlen+1) * (qlen+1) * sizeof(uint8_t), cudaHostAllocDefault);
    s_args[i].row0_d = (int *) s_args[i].mem;
    s_args[i].row1_d = s_args[i].row0_d + (tlen + 1);
    s_args[i].row2_d = s_args[i].row1_d + (tlen + 1);
    s_args[i].mat_d = (uint8_t *) (s_args[i].row2_d + (tlen + 1));
    s_args[i].t_d = (char *) (s_args[i].mat_d + (tlen + 1) * (qlen + 1));
    s_args[i].q_d = s_args[i].t_d + tlen;
  }

  // Launch depth first kernel streams.
  for (size_t i = 0; i < batch_size; ++i) {
    cudaMemcpyAsync(s_args[i].t_d, tests[i].s1, tests[i].s1_len * sizeof(char), cudaMemcpyHostToDevice, s[i]);
    cudaMemcpyAsync(s_args[i].q_d, tests[i].s2, tests[i].s2_len * sizeof(char), cudaMemcpyHostToDevice, s[i]);
    needletail_init_kernel <<< divide_then_round_up((tests[i].s1_len + 1), BLOCK_SIZE), BLOCK_SIZE, 0, s[i] >>>
        (tests[i].s1_len, tests[i].s2_len, GAP_SCORE, s_args[i].row0_d, s_args[i].row1_d, s_args[i].mat_d);
    needletail_comp_kernel <<< 1, BLOCK_SIZE, 0, s[i] >>>
      (s_args[i].t_d, s_args[i].q_d, tests[i].s1_len, tests[i].s2_len, GAP_SCORE,
        s_args[i].row0_d, s_args[i].row1_d, s_args[i].row2_d, s_args[i].mat_d);
    cudaMemcpyAsync(s_args[i].mat, s_args[i].mat_d, (tests[i].s1_len + 1) * (tests[i].s2_len + 1) * sizeof(uint8_t), cudaMemcpyDeviceToHost, s[i]);
  }

  // Release the scheduling lock.
  mtx.unlock();

  for (size_t i = 0; i < batch_size; ++i) {
    cudaStreamSynchronize(s[i]);
    cudaFree(s_args[i].mem);
    cudaStreamDestroy(s[i]);
    std::pair<char *, char *> algn = nw_ptr_backtrack(s_args[i].mat,
      swap_t_q[i], tests[i].s1, tests[i].s2, tests[i].s1_len, tests[i].s2_len);
    cudaFreeHost(s_args[i].mat);
    test_batch.log_result(tests[i].id, algn.first, algn.second, 0, 0);
  }

  return;
}

#endif