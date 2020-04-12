#ifndef NEEDLETAIL_KERNELS_CUH
#define NEEDLETAIL_KERNELS_CUH

#include "nw_general.h"
#include "testbatch.hpp"
#include "cuda_error_check.cuh"

__global__ void needletail_init_kernel (
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

__global__ void needletail_comp_kernel (
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

#endif