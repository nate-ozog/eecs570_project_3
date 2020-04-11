#ifndef XS_CORE_CUH
#define XS_CORE_CUH

#include "nw_general.h"
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

std::pair<uint8_t *, int> xs_t_geq_q_man(
  const char * t,
  const char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  void * mem
) {

  // Maintain a sliding window of 3 rows of our transformed matrix.
  // This is useful because with the transformation matrix we get
  // complete memory coalescing on both reads and writes.
  int * row0_d = (int *) mem;
  int * row1_d = row0_d + (tlen + 1);
  int * row2_d = row1_d + (tlen + 1);

  // Maintain a full untransformed matrix (of back-pointers) for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  uint8_t * mat_d = (uint8_t *) (row2_d + (tlen + 1));

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cuda_error_check(cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice));
  cuda_error_check(cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice));

  // Prepare the first 2 rows of our transformed compute matrix,
  // and the border elements for our untranformed matrix.
  dim3 init_grid_dim(divide_then_round_up((tlen + 1), 1024), 1, 1);
  dim3 init_block_dim(1024, 1, 1);
  xs_core_init <<<init_grid_dim, init_block_dim>>>
    (tlen, qlen, mis_or_ind, row0_d, row1_d, mat_d);

  // Launch our dynamic programming kernel.
  dim3 comp_grid_dim(1, 1, 1);
  dim3 comp_block_dim(1024, 1, 1);
  xs_core_comp <<< comp_grid_dim, comp_block_dim>>>
    (t_d, q_d, tlen, qlen, mis_or_ind, row0_d,
      row1_d, row2_d, mat_d);

  // Allocate pinned memory on the host for faster data transfer.
  uint8_t * mat;
  size_t cpu_ptr_mat_bytes = (tlen+1) * (qlen+1) * sizeof(uint8_t);
  cuda_error_check(cudaHostAlloc((void**) & mat, cpu_ptr_mat_bytes, cudaHostAllocDefault));

  // Copy back our untransformed matrix to the host.
  int opt_score = 0;
  cuda_error_check(cudaMemcpy(mat, mat_d, cpu_ptr_mat_bytes, cudaMemcpyDeviceToHost));
  cuda_error_check(cudaMemcpy(&opt_score, &row1_d[tlen], sizeof(int), cudaMemcpyDeviceToHost));

  std::pair<uint8_t *, int> res (mat, opt_score);
  return res;
}

#endif