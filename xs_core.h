#ifndef XS_CORE_CUH
#define XS_CORE_CUH

#include "nw_general.h"
#include "cuda_error_check.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;


__global__ void xs_core_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * xf_mat_row0,
  int * xf_mat_row1,
  uint8_t * mat
) {
  // Get the global thread index.
  uint32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Initialize left column of backtrack matrix
  if (g_tx < qlen + 1)
	  mat[g_tx*(tlen+1)] = INS;
  // Initialize top row of backtrack matrix
  if (g_tx < tlen + 1)
    mat[g_tx] = DEL;
  // Write 0 to the first cell of our transformed matrix row0.
  if (g_tx == 0)
    xf_mat_row0[0] = 0;
  // Write g_tx * mis_or_ind to the first and
  // second cell of the tranformed matrix row1.
  if (g_tx == 1) {
    xf_mat_row1[0] = g_tx * mis_or_ind;
    xf_mat_row1[1] = g_tx * mis_or_ind;
  }
}


__global__ void xs_core_comp(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  int * xf_mat_row0,
  int * xf_mat_row1,
  int * xf_mat_row2,
  signed char mis_or_ind,
  uint8_t * mat,
  uint32_t nblocks
) {
  // Get the global and local thread index.
  int * xf_mat_row_temp = NULL;
  uint32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint32_t l_tx = threadIdx.x;
  grid_group grid = this_grid();
  extern __shared__ int smem[];
  int * s_row_up = smem;

  // Kernel management variables, see kernel definiton
  // for comments on what these are used for.
  bool wr_q_border_elt = true;
  bool wr_t_border_elt = true;
  uint32_t comp_w = 0;
  uint32_t comp_x_off = 1;

  // Other variables to help manage the kernel manager variables...
  bool square_matrix = (qlen == tlen);
  uint32_t max_comp_w_cnt = 0;
  int8_t comp_w_increment = 1;
  uint32_t max_comp_w = qlen < tlen ? qlen : tlen;
  uint32_t largest_dim = tlen > qlen ? tlen : qlen;
  uint32_t smallest_dim = tlen < qlen ? tlen : qlen;
  uint32_t max_comp_w_cnt_max = largest_dim - smallest_dim + 1;

  // Loop through every wavefront/diagonal.
  for (uint32_t comp_y_off = 2; comp_y_off < qlen + tlen + 1; ++comp_y_off) {
    // Update kernel management variables.
    wr_q_border_elt = (comp_y_off < qlen + 1);
    wr_t_border_elt = (comp_y_off < tlen + 1);
    comp_w += comp_w_increment;
    comp_x_off += (comp_y_off > qlen + 1);

	  // If we need to write a border element for our query.
	  if (g_tx == 0 && wr_q_border_elt)
		xf_mat_row2[0] = (comp_y_off) * mis_or_ind;
	  // If we need to write a border element for our target.
	  if (g_tx == 0 && wr_t_border_elt)
		xf_mat_row2[comp_x_off + comp_w] = (comp_y_off) * mis_or_ind;
	  // If we are in the compute region.
	  if (g_tx >= comp_x_off && g_tx < comp_x_off + comp_w) {
		// Fetch into shared memory.
		if (l_tx == 0 || g_tx == comp_x_off)
		  s_row_up[l_tx] = xf_mat_row1[g_tx - 1];
		s_row_up[l_tx + 1] = xf_mat_row1[g_tx];
	  }
	  __syncthreads();
	  if (g_tx >= comp_x_off && g_tx < comp_x_off + comp_w) {
		// Do the NW cell calculation.
		int match = xf_mat_row0[g_tx - 1]
		  + cuda_nw_get_sim(q[comp_y_off - g_tx - 1], t[g_tx - 1]);
		int ins = s_row_up[l_tx + 1] + mis_or_ind;
		int del = s_row_up[l_tx] + mis_or_ind;
		// Write back to our current sliding window row index, set pointer.
		int mat_idx = g_tx + (comp_y_off-g_tx)*(tlen+1);
		if (match >= ins && match >= del) {
			xf_mat_row2[g_tx] = match;
			mat[mat_idx] = MATCH;
		}
		else if (ins >= match && ins >= del) {
			xf_mat_row2[g_tx] = ins;
			mat[mat_idx] = INS;
		}
		else {
			xf_mat_row2[g_tx] = del;
			mat[mat_idx] = DEL;
		}
	  }

    // Update other management variables.
    max_comp_w_cnt = comp_w == max_comp_w ?
      max_comp_w_cnt += 1 : max_comp_w_cnt;
    if (square_matrix)
      comp_w_increment = max_comp_w_cnt == 1 ? -1 : 1;
    else
      comp_w_increment = max_comp_w_cnt == 0 ? 1 :
        max_comp_w_cnt == max_comp_w_cnt_max ? -1 : 0;

    // Slide our window in our compute matrix.
    xf_mat_row_temp = xf_mat_row0;
    xf_mat_row0 = xf_mat_row1;
    xf_mat_row1 = xf_mat_row2;
    xf_mat_row2 = xf_mat_row_temp;

	grid.sync();
  }
}

uint8_t * xs_t_geq_q_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  void * mem,
  cudaStream_t * stream
) {

  // Maintain a sliding window of 3 rows of our transformed matrix.
  // This is useful because with the transformation matrix we get
  // complete memory coalescing on both reads and writes.
  int * xf_mat_row0_d = (int *) mem;
  int * xf_mat_row1_d = xf_mat_row0_d + (tlen + 1);
  int * xf_mat_row2_d = xf_mat_row1_d + (tlen + 1);

  // Maintain a full untransformed matrix (of back-pointers) for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  uint8_t * mat_d = (uint8_t *) (xf_mat_row2_d + (tlen + 1));

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, *stream);

  // Prepare the first 2 rows of our transformed compute matrix,
  // and the border elements for our untranformed matrix.
  uint32_t init_num_threads = (tlen + 1) > (qlen + 1) ? (tlen + 1) : (qlen + 1);
  dim3 init_g_dim(ceil(init_num_threads / ((float) 1024)));
  dim3 init_b_dim(1024);
  xs_core_init <<<init_g_dim, init_b_dim, 0, *stream>>>
    (tlen, qlen, mis_or_ind, xf_mat_row0_d, xf_mat_row1_d, mat_d);

  // Run our matrix scoring algorithm.
  uint32_t comp_num_threads = (tlen + 1);
  float threads_per_block = 1024;
  uint32_t nblocks = ceil(comp_num_threads / threads_per_block);
  dim3 comp_g_dim(nblocks, 1, 1);
  dim3 comp_b_dim(threads_per_block, 1, 1);

  // Launch our kernel.
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if (!deviceProp.cooperativeLaunch) {
	  printf("Device does not support cooperative groups!\n");
	  exit(0);
  }

  void *kernelArgs[] = {
	  (void*) &t_d, (void*) &q_d, (void*) &tlen, (void*) &qlen,
	  (void*) &xf_mat_row0_d, (void*) &xf_mat_row1_d, (void*) &xf_mat_row2_d,
	  (void*) &mis_or_ind, (void*) &mat_d, (void*) &nblocks
  };
  cuda_error_check( cudaLaunchCooperativeKernel(
		  (void*) xs_core_comp,
		  comp_g_dim,
		  comp_b_dim,
		  kernelArgs,
		  (threads_per_block+1)*sizeof(int),
		  *stream));

  // Allocate pinned memory on the host for faster data transfer.
  uint8_t* mat;
  size_t cpu_ptr_mat_bytes = (tlen+1) * (qlen+1) * sizeof(uint8_t);
  cuda_error_check( 
		  cudaHostAlloc((void**)&mat, cpu_ptr_mat_bytes, cudaHostAllocDefault));

  // Copy back our untransformed matrix to the host.
  cuda_error_check( 
		  cudaMemcpyAsync(mat, mat_d, cpu_ptr_mat_bytes, 
			  cudaMemcpyDeviceToHost, *stream) );

  return mat;
}

#endif
