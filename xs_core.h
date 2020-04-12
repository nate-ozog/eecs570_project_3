#ifndef XS_CORE_CUH
#define XS_CORE_CUH

#include "nw_general.h"
#include "cuda_error_check.h"

__global__ void xs_core_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
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
	mat[0] = 0;
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
	// Shorthand thread index and block dimensions.
	uint32_t tidx = threadIdx.x;
	uint32_t bdim = blockDim.x;

	// Set up shared memory row pointers.
	extern __shared__ int smem[];
	int * s_row0 = smem;
	int * s_row1 = s_row0 + (bdim+1)*max_strides;
	int * s_row2 = s_row1 + (bdim+1)*max_strides;
	int * s_temp = NULL;

	// Initialize Shared Memory (top two rows of sheared matrix section).
	if (tidx == 0) {
		s_row0[0] = col[col_offset-1];		 // on diag, optimize later
		s_row1[0] = col[col_offset];		 // passed from previous kernel
		s_row1[1] = col_offset * mis_or_ind; // always on diag
	}
	__syncthreads();

	// Loop through every row that has useful work.
	for (uint32_t row_idx = col_offset+1; row_idx < qlen+tlen+1; row_idx++) {

		// Calculate row offset, due to diagonalization past row tlen.
		uint32_t row_offset = row_idx <= tlen ? 0 : row_idx - (tlen+1);

		// Calculate last column computed by kernel (local and global).
		uint32_t last_gcol = min(qlen, col_offset+max_strides*bdim-1);
		uint32_t last_lcol = min(max_strides*bdim-1, last_gcol-col_offset);

		// Set Shared Memory values for leftmost column and diagonal
		if (tidx == 0) {
			s_row2[0] = col[row_idx];
			if (row_idx <= last_gcol)
				s_row2[row_idx-col_offset+1] = row_idx * mis_or_ind;
		}

		// Stride across columns.
		for (uint32_t col_idx = 0; col_idx <= last_lcol; col_idx += bdim) {

			// Calculate global and local indices for current thread.
			uint32_t gidx = col_offset + col_idx + tidx;
			uint32_t lidx = col_idx + tidx;

			// Do the NW cell calculation.
			if (gidx > row_offset && gidx <= last_gcol) {
				int match = s_row0[lidx] + cuda_nw_get_sim(q[gidx-1], t[row_idx-gidx-1]);
				int del = s_row1[lidx+1] + mis_or_ind;
				int ins = s_row1[lidx] + mis_or_ind;

				// Write back to our current sliding window row index, set pointer.
				int mat_idx = row_idx-gidx + gidx*(tlen+1);
				if (match >= ins && match >= del) {
					s_row2[lidx+1] = match;
					mat[mat_idx] = MATCH;
					if (lidx == last_lcol)
						col[row_idx] = match;
				}
				else if (ins >= match && ins >= del) {
					s_row2[lidx+1] = ins;
					mat[mat_idx] = INS;
					if (lidx == last_lcol)
						col[row_idx] = ins;
				}
				else {
					s_row2[lidx+1] = del;
					mat[mat_idx] = DEL;
					if (lidx == last_lcol)
						col[row_idx] = del;
				}
			}
		}

		// Shift sliding window.
		s_temp = s_row0;
		s_row0 = s_row1;
		s_row1 = s_row2;
		s_row2 = s_temp;
		__syncthreads();
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

  // Save last column that the kernel computes, and initialize.
  int * col = new int [tlen+qlen+1];
  for(int i = 0; i < tlen+qlen+1; i++)
	  col[i] = i * mis_or_ind;
  int * col_d = (int *) mem;
  cudaMemcpyAsync(col_d, col, (tlen+qlen+1)*sizeof(int), cudaMemcpyHostToDevice, *stream);

  // Maintain a full untransformed matrix (of back-pointers) for PCIe transfer after
  // compute is done. This min/maxes our memory utilization.
  uint8_t * mat_d = (uint8_t *) (col_d + (tlen + qlen + 1));

  // Pointers to target and query.
  char * t_d = (char *) (mat_d + (tlen + 1) * (qlen + 1));
  char * q_d = t_d + tlen;

  // Copy our target and query to the GPU.
  cudaMemcpyAsync(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice, *stream);

  // Set the border elements of our compressed matrix.
  dim3 init_grid_dim(CEILDIV(tlen+1, 1024));
  dim3 init_block_dim(1024);
  xs_core_init <<<init_grid_dim, init_block_dim, 0, *stream>>>
    (tlen, qlen, mis_or_ind, mat_d);

  // Maximize SM and Shared Mem utilization
  uint32_t block_size = 1024;
  uint32_t max_strides = 5;

  // Loop through every wavefront/diagonal.
  for (uint32_t col_idx = 1; col_idx < qlen+1; col_idx+=max_strides*block_size) {
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

  delete [] col;

  return mat;
}

#endif
