#ifndef NEEDLETAIL_THREADS_CUH
#define NEEDLETAIL_THREADS_CUH

#include "nw_general.h"
#include "testbatch.hpp"
#include "cuda_error_check.cuh"
#include "needletail_kernels.cuh"

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

struct d2h_staller_args {
  stream_args ** s_args;
  cudaStream_t ** s;
  Test_t ** tests;
};

struct bt_worker_args {
  Test_t * test;
  uint8_t * mat;
};

void * bt_worker (void * arg) {
  struct bt_worker_args * args = (struct bt_worker_args *) arg;
  Test_t * test = args->test;
  uint8_t * mat = args->mat;
  std::pair<char *, char *> algn = nw_ptr_backtrack(mat, 0, test->s1, test->s2, test->s1_len, test->s2_len);
  test_batch.log_result(test->id, algn.first, algn.second, 0, 0);
  delete [] algn.first;
  delete [] algn.second;
  cudaFreeHost(mat);
  delete test;
  __atomic_sub_fetch(&oustanding_batches, 1, __ATOMIC_RELAXED);
  pthread_exit(NULL);
}

void * d2h_staller (void * arg) {
  d2h_staller_args * args = (struct d2h_staller_args *) arg;
  for (size_t i = 0; i < STREAM_BATCH_SIZE; ++i) {
    cudaStreamSynchronize(*args->s[i]);
    cudaFree(args->s_args[i]->mat);
    cudaStreamDestroy(*args->s[i]);
    pthread_t bt_worker_thread;
    bt_worker_args * bt_args = new bt_worker_args;
    bt_args->test = args->tests[i];
    bt_args->mat = args->s_args[i]->mat;
    delete args->s[i];
    delete args->s_args[i];
    pthread_create(&bt_worker_thread, NULL, bt_worker, bt_args);
    pthread_detach(bt_worker_thread);
  }
  delete args;
  pthread_exit(NULL);
}

void needletail_stream_batch (
  Test_t ** tests
) {

  // Create an array of stream arguments and stream pointers.
  stream_args ** s_args = new stream_args * [STREAM_BATCH_SIZE];
  cudaStream_t ** s = new cudaStream_t * [STREAM_BATCH_SIZE];

  // Create the streams.
  for (size_t i = 0; i < STREAM_BATCH_SIZE; ++i) {
    s[i] = new cudaStream_t;
    cudaStreamCreate(s[i]);
  }

  // Claim GPU memory and setup arguments.
  for (size_t i = 0; i < STREAM_BATCH_SIZE; ++i) {
    uint32_t tlen = tests[i]->s1_len;
    uint32_t qlen = tests[i]->s2_len;
    uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
    num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(uint8_t);
    num_GPU_mem_bytes += tlen * sizeof(char);
    num_GPU_mem_bytes += qlen * sizeof(char);
    s_args[i] = new stream_args;
    cudaMalloc((void **) & s_args[i]->mem, num_GPU_mem_bytes);
    cudaHostAlloc((void**)& s_args[i]->mat, (tlen+1) * (qlen+1) * sizeof(uint8_t), cudaHostAllocDefault);
    s_args[i]->row0_d = (int *) s_args[i]->mem;
    s_args[i]->row1_d = s_args[i]->row0_d + (tlen + 1);
    s_args[i]->row2_d = s_args[i]->row1_d + (tlen + 1);
    s_args[i]->mat_d = (uint8_t *) (s_args[i]->row2_d + (tlen + 1));
    s_args[i]->t_d = (char *) (s_args[i]->mat_d + (tlen + 1) * (qlen + 1));
    s_args[i]->q_d = s_args[i]->t_d + tlen;
  }

  // Launch depth first kernel streams.
  for (size_t i = 0; i < STREAM_BATCH_SIZE; ++i) {
    cudaMemcpyAsync(s_args[i]->t_d, tests[i]->s1, tests[i]->s1_len * sizeof(char), cudaMemcpyHostToDevice, *s[i]);
    cudaMemcpyAsync(s_args[i]->q_d, tests[i]->s2, tests[i]->s2_len * sizeof(char), cudaMemcpyHostToDevice, *s[i]);
    needletail_init_kernel <<< divide_then_round_up((tests[i]->s1_len + 1), BLOCK_SIZE), BLOCK_SIZE, 0, *s[i] >>>
        (tests[i]->s1_len, tests[i]->s2_len, GAP_SCORE, s_args[i]->row0_d, s_args[i]->row1_d, s_args[i]->mat_d);
    needletail_comp_kernel <<< 1, BLOCK_SIZE, 0, *s[i] >>>
      (s_args[i]->t_d, s_args[i]->q_d, tests[i]->s1_len, tests[i]->s2_len, GAP_SCORE,
        s_args[i]->row0_d, s_args[i]->row1_d, s_args[i]->row2_d, s_args[i]->mat_d);
    cudaMemcpyAsync(s_args[i]->mat, s_args[i]->mat_d, (tests[i]->s1_len + 1) * (tests[i]->s2_len + 1) * sizeof(uint8_t), cudaMemcpyDeviceToHost, *s[i]);
  }

  // Peel off a thread to handle stream completion.
  pthread_t d2h_staller_thread;
  d2h_staller_args * d2h_args = new d2h_staller_args;
  d2h_args->s_args = s_args;
  d2h_args->s = s;
  d2h_args->tests = tests;
  pthread_create(&d2h_staller_thread, NULL, d2h_staller, d2h_args);
  pthread_detach(d2h_staller_thread);
  oustanding_batches += STREAM_BATCH_SIZE;
  return;
}

#endif