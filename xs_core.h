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
  uint32_t row,
  uint32_t g_offset,
  uint32_t g_nthreads,
  signed char mis_or_ind,
  int * row0,
  int * row1,
  int * row2,
  uint8_t * mat
) {
  uint32_t g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t g_idx = g_tid + g_offset;
  uint32_t l_tid = threadIdx.x;
  uint32_t l_offset = max(0, g_offset - blockIdx.x * blockDim.x);
  uint32_t l_idx = l_tid + l_offset;
  extern __shared__ int smem[];
  int * s_row_up = smem;

  // Fill in SM at computed positions
  /* if (g_tid < g_nthreads) */
	  /* s_row_up[l_idx] = row1[g_idx]; */
  
  // If we need to write a border element on diagonal
  if (g_tid == 0 && row <= qlen)
		row2[row] = row * mis_or_ind;

  // If we need to write a border element in left column
  if (l_tid == 0 && row <= tlen) {
    /* s_row_up[0] = row1[0]; // Also write in SM border */
	if (g_tid == 0)
		row2[0] = row * mis_or_ind;
  }

  // Synchronize all threads, so that SM values are set.
  __syncthreads();

  // Do the NW cell calculation.
  if (g_tid < g_nthreads) {
    int match = row0[g_idx-1] + cuda_nw_get_sim(q[g_idx-1], t[row-g_idx-1]);
    /* int del = s_row_up[l_idx] + mis_or_ind; */
    /* int ins = s_row_up[l_idx-1] + mis_or_ind; */
	int del = row1[g_idx] + mis_or_ind;
	int ins = row1[g_idx-1] + mis_or_ind;

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


uint32_t get_nthreads(uint32_t row, uint32_t tlen, uint32_t qlen)
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

  // Maintain a sliding window of 3 rows of our transformed matrix.
  // This is useful because with the transformation matrix we get
  // complete memory coalescing on both reads and writes.
  int * row_temp = NULL;
  int * row0_d = (int *) mem;
  int * row1_d = row0_d + (qlen + 1);
  int * row2_d = row1_d + (qlen + 1);

  // Maintain a full untransformed matrix (of back-pointers) for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  uint8_t * mat_d = (uint8_t *) (row2_d + (qlen + 1));

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, *stream);

  // Prepare the first 2 rows of our transformed compute matrix,
  // and the border elements for our untranformed matrix.
  uint32_t init_nthreads = (tlen + 1) > (qlen + 1) ? (tlen + 1) : (qlen + 1);
  dim3 init_grid_dim(ceil(init_nthreads / ((float) 1024)));
  dim3 init_block_dim(1024);
  xs_core_init <<<init_grid_dim, init_block_dim, 0, *stream>>>
    (tlen, qlen, mis_or_ind, row0_d, row1_d, mat_d);

  // Loop through every wavefront/diagonal.
  for (uint32_t row = 2; row < qlen+tlen+1; ++row) {

	  // Calculate grid size for this row.
	  uint32_t block_size = 1024;
	  uint32_t nthreads = get_nthreads(row, tlen, qlen);
	  uint32_t offset = get_offset(row, tlen, qlen);
	  dim3 grid_dim(ceil(nthreads / ((float) block_size)));

    // Launch our kernel.
    xs_core_comp <<< grid_dim, block_size, block_size * sizeof(int), *stream >>>
      (t_d, q_d, tlen, qlen, row, offset, nthreads, 
	    mis_or_ind, row0_d, row1_d, row2_d, mat_d);

    // Slide our window in our compute matrix.
    row_temp = row0_d;
    row0_d = row1_d;
    row1_d = row2_d;
    row2_d = row_temp;
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
