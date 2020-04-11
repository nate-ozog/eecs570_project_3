#ifndef XS_CUH
#define XS_CUH

#include "nw_general.h"
#include "testbatch.hpp"
#include "cuda_error_check.cuh"

__global__ void xs_core_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * row0,
  int * row1,
  uint8_t * mat
) {
  uint32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tx < tlen + 1)
    mat[tx] = DEL;
  if (tx < qlen + 1)
    mat[tx * (tlen + 1)] = INS;
  if (tx == 0) {
    row0[0] = 0;
    mat[0] = 0;
  }
  if (tx < 2)
    row1[tx] = mis_or_ind;
}

__global__ void xs_core_comp(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * row0,
  int * row1,
  int * row2,
  uint8_t * mat
) {
  uint32_t tx = threadIdx.x;
  int * rowt = NULL;
  // __shared__ int s_row1[1025];

  // Kernel management variables, see kernel definiton
  // for comments on what these are used for.
  bool wr_q_border_elt = true;
  bool wr_t_border_elt = true;
  uint32_t comp_w = 0;
  uint32_t comp_x_off = 1;

  // Other variables to help manage the kernel manager variables...
  uint32_t max_comp_w_cnt = 0;
  int8_t comp_w_increment = 1;
  uint32_t max_comp_w = qlen < tlen ? qlen : tlen;
  uint32_t max_comp_w_cnt_max = tlen - qlen + 1;

  // Loop through every wavefront/diagonal.
  for (uint32_t i = 2; i < qlen + tlen + 1; ++i) {

    // Update kernel management variables.
    wr_q_border_elt = (i < qlen + 1);
    wr_t_border_elt = (i < tlen + 1);
    comp_w += comp_w_increment;
    comp_x_off += (i > qlen + 1);
    // If we need to write a border element for our query.
    if (tx == 0 && wr_q_border_elt)
      row2[0] = i * mis_or_ind;
    // If we need to write a border element for our target.
    if (tx == 0 && wr_t_border_elt)
      row2[comp_x_off + comp_w] = i * mis_or_ind;

    for (uint32_t j = comp_x_off; j < comp_x_off + comp_w; j += 1024) {
      uint32_t comp_idx = j + tx;

      // if (comp_idx < comp_x_off + comp_w) {
      //   if (tx == 0)
      //     s_row1[tx] = row1[comp_idx - 1];
      //   s_row1[tx + 1] = row1[comp_idx];
      // }

      __syncthreads();

      if (comp_idx < comp_x_off + comp_w) {
        int match = row0[comp_idx - 1]
          + cuda_nw_get_sim(q[i - comp_idx - 1], t[comp_idx - 1]);
        int ins = row1[comp_idx] + mis_or_ind;
        int del = row1[comp_idx - 1] + mis_or_ind;
        int ptr_idx = (tlen + 1) * (i - comp_idx) + comp_idx;
        if (match >= ins && match >= del) {
          row2[comp_idx] = match;
          mat[ptr_idx] = MATCH;
        }
        else if (ins >= match && ins >= del) {
          row2[comp_idx] = ins;
          mat[ptr_idx] = INS;
        }
        else {
          row2[comp_idx] = del;
          mat[ptr_idx] = DEL;
        }
      }

    }

    // Update other management variables.
    max_comp_w_cnt = comp_w == max_comp_w ?
      max_comp_w_cnt += 1 : max_comp_w_cnt;
    if (qlen == tlen)
      comp_w_increment = max_comp_w_cnt == 1 ? -1 : 1;
    else
      comp_w_increment = max_comp_w_cnt == 0 ? 1 :
        max_comp_w_cnt == max_comp_w_cnt_max ? -1 : 0;

    // Slide our window.
    rowt = row0;
    row0 = row1;
    row1 = row2;
    row2 = rowt;
  }
}

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

void stream_man (
  Test_t tests[NUM_HW_QS]
) {

  stream_args s_args[NUM_HW_QS];
  cudaStream_t streams[NUM_HW_QS];

  for (size_t i = 0; i < NUM_HW_QS; ++i) {
    cudaStreamCreate(&streams[i]);
    uint32_t tlen = tests[i].s1_len;
    uint32_t qlen = tests[i].s2_len;
    uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
    num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(uint8_t);
    num_GPU_mem_bytes += tlen * sizeof(char);
    num_GPU_mem_bytes += qlen * sizeof(char);
    cudaMalloc((void **) & s_args[i].mem, num_GPU_mem_bytes);
    s_args[i].mat = new uint8_t[(tlen + 1) * (qlen + 1)];
    s_args[i].row0_d = (int *) s_args[i].mem;
    s_args[i].row1_d = s_args[i].row0_d + (tlen + 1);
    s_args[i].row2_d = s_args[i].row1_d + (tlen + 1);
    s_args[i].mat_d = (uint8_t *) (s_args[i].row2_d + (tlen + 1));
    s_args[i].t_d = (char *) (s_args[i].mat_d + (tlen + 1) * (qlen + 1));
    s_args[i].q_d = s_args[i].t_d + tlen;
  }

  for (size_t i = 0; i < NUM_HW_QS; ++i) {
    cudaMemcpyAsync(s_args[i].t_d, tests[i].s1, tests[i].s1_len * sizeof(char), cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(s_args[i].q_d, tests[i].s2, tests[i].s2_len * sizeof(char), cudaMemcpyHostToDevice, streams[i]);
  }

  for (size_t i = 0; i < NUM_HW_QS; ++i) {
    xs_core_init <<<divide_then_round_up((tests[i].s1_len + 1), 1024), BLOCK_SIZE, 0, streams[i]>>>
      (tests[i].s1_len, tests[i].s2_len, GAP_SCORE, s_args[i].row0_d, s_args[i].row1_d, s_args[i].mat_d);
    xs_core_comp <<< 1, BLOCK_SIZE, 0, streams[i]>>>
      (s_args[i].t_d, s_args[i].q_d, tests[i].s1_len, tests[i].s2_len, GAP_SCORE,
        s_args[i].row0_d, s_args[i].row1_d, s_args[i].row2_d, s_args[i].mat_d);
  }

  for (size_t i = 0; i < NUM_HW_QS; ++i)
    cudaMemcpyAsync(s_args[i].mat, s_args[i].mat_d, (tests[i].s1_len + 1) * (tests[i].s2_len + 1) * sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[i]);

  // SET BOOL.

  for (size_t i = 0; i < NUM_HW_QS; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaFree(s_args[i].mem);
    delete [] s_args[i].mat;
    cudaStreamDestroy(streams[i]);
  }

  return;
}

#endif