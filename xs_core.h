#ifndef XS_CORE_CUH
#define XS_CORE_CUH

#include "nw_general.h"
#include "cuda_error_check.h"

__global__ void xs_core_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * row0,
  int * row1,
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
  if (g_tx == 0) {
    row0[0] = 0;
	mat[0] = 0;
  }
  // Write g_tx * mis_or_ind to the first and
  // second cell of the tranformed matrix row1.
  if (g_tx < 2)
    row1[g_tx] = mis_or_ind;
}



__global__ void xs_core_comp(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  int * col,
  uint32_t col_offset,
  uint32_t max_strides,
  signed char mis_or_ind,
  uint8_t * mat
) {
  // Set up shared memory row pointers.
  extern __shared__ int smem[];
  int * s_row0 = smem;
  int * s_row1 = s_row0 + (blockDim.x+1)*max_strides;
  int * s_row2 = s_row1 + (blockDim.x+1)*max_strides;
  int * s_temp = NULL;

  //
  uint32_t tid = threadIdx.x;
  uint32_t tidx = tid + g_offset;

  // Loop through every row.
  for (uint32_t row_idx = 2; row_idx < qlen+tlen+1; row_idx++) {

	  uint32_t band_width = get_band_width(row, tlen, qlen);
	  uint32_t offset = get_offset(row, tlen, qlen);

	  // 
	  for (uint32_t stride_idx = 0; stride_idx < max_strides; stride_idx++) {
		  // Fill in SM at computed positions
		  if (g_tid < g_nthreads)
			  s_row_up[l_tid+1] = row1[g_idx];
		  
		  // Write SM border element
		  if (l_tid == 0)
			  s_row_up[0] = row1[g_idx-1];

		  // If we need to write a border element on diagonal
		  if (g_tid == 0 && row <= qlen)
			  row2[row] = row * mis_or_ind;

		  // If we need to write a border element in left column
		  if (g_tid == 0 && row <= tlen)
			  row2[0] = row * mis_or_ind;

		  // Synchronize all threads, so that SM values are set.
		  __syncthreads();

		  // Do the NW cell calculation.
		  if (g_tid < g_nthreads) {
			int match = row0[g_idx-1] + cuda_nw_get_sim(q[g_idx-1], t[row-g_idx-1]);
			int del = s_row_up[l_tid+1] + mis_or_ind;
			int ins = s_row_up[l_tid] + mis_or_ind;

			// Write back to our current sliding window row index, set pointer.
			int mat_idx = row-g_idx + g_idx*(tlen+1);
			if (match >= ins && match >= del) {
				row2[g_idx] = match;
				mat[mat_idx] = MATCH;
			}
			else if (ins >= match && ins >= del) {
				row2[g_idx] = ins;
				mat[mat_idx] = INS;
			}
			else {
				row2[g_idx] = del;
				mat[mat_idx] = DEL;
			}
		  }
	  }

	  // Shift sliding window.
	  s_temp = s_row0;
	  s_row0 = s_row1;
	  s_row1 = s_row2;
	  s_row2 = s_temp;
  }
}


uint32_t get_band_width(uint32_t row, uint32_t tlen, uint32_t qlen)
{ // we can assume tlen >= qlen
	if (row < 2) // no work
		return 0;
	else if (row <= qlen)
		return row - 1;
	else if (row <= tlen)
		return qlen;
	else
		return qlen+tlen+1 - row;
}


uint32_t get_offset(uint32_t row, uint32_t tlen, uint32_t qlen)
{ // we can assume tlen >= qlen
	if (row < 2) // invalid, shouldn't happen
		return 0;
	else if (row <= tlen)
		return 1;
	else
		return row - tlen;
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

  // Save last column that the kernel computes
  int * col_d = (int *) mem;

  // Maintain a full untransformed matrix (of back-pointers) for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  uint8_t * mat_d = (uint8_t *) (col_d + (tlen + qlen + 1));

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, *stream);

  // Prepare the first 2 rows of our transformed compute matrix,
  // and the border elements for our untranformed matrix.
  uint32_t init_nthreads = (tlen + 1) > (qlen + 1) ? (tlen + 1) : (qlen + 1);
  dim3 init_grid_dim(CEILDIV(init_nthreads, 1024));
  dim3 init_block_dim(1024);
  xs_core_init <<<init_grid_dim, init_block_dim, 0, *stream>>>
    (tlen, qlen, mis_or_ind, row0_d, row1_d, mat_d);

  // Maximize SM and Shared Mem utilization
  uint32_t block_size = 1024;
  uint32_t max_strides = 5;

  // Loop through every wavefront/diagonal.
  for (uint32_t col_idx = 0; col_idx < qlen+1; col_idx+=max_strides*block_size) {
	uint32_t shared_mem_size = (block_size+1) * sizeof(int) * max_strides * 3;
    xs_core_comp <<< 1, block_size, shared_mem_size, *stream >>>
      (t_d, q_d, tlen, qlen, col_d, col_idx, max_strides, mis_or_ind, mat_d);
  }

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
