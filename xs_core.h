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
	// Set up shared memory row pointers.
	extern __shared__ int smem[];
	int * s_row0 = smem;
	int * s_row1 = s_row0 + (blockDim.x+1)*max_strides;
	int * s_row2 = s_row1 + (blockDim.x+1)*max_strides;
	int * s_temp = NULL;

	// Shorthand thread index.
	uint32_t tidx = threadIdx.x;

	// Initialize Shared Memory (top two rows of sheared matrix).
	if (tidx == 0) {
		s_row0[0] = col[col_offset-1];
		s_row1[0] = col[col_offset];
		s_row1[1] = col_offset * mis_or_ind;
	}
	__syncthreads();

	// Loop through every row that has useful work.
	for (uint32_t row_idx = col_offset+1; row_idx < qlen+tlen+1; row_idx++) {

		// Calculate full sheared matrix band width based on row.
		uint32_t band_width;
		if (row_idx <= qlen)
			band_width = row_idx - 1;
		else if (row_idx <= tlen)
			band_width = qlen;
		else
			band_width = qlen+tlen+1 - row_idx;

		// Calculate offset based on row in sheared matrix.
		uint32_t row_offset = row_idx <= tlen ? 0 : row_idx-tlen;
		uint32_t offset = row_offset + col_offset;

		// Set Shared Memory (leftmost col and diag of current row).
		if (tidx == 0) {
			s_row2[0] = col[row_idx];
			if (row_idx <= qlen) {
				s_row2[row_idx-col_offset+1] = row_idx * mis_or_ind;
			}
		}

		// Stride across columns, taking offsets into account
		for (uint32_t col_idx = 0; 
				col_idx < band_width-col_offset+1 && 
				col_idx < max_strides*blockDim.x; 
				col_idx += blockDim.x) {

			// Calculate global and local indices for current thread.
			uint32_t gidx = col_idx + offset + tidx;
			uint32_t lidx = col_idx + tidx;

			// Do the NW cell calculation.
			if (gidx < band_width) {
				int match = s_row0[lidx-1] + cuda_nw_get_sim(q[gidx-1], t[row_idx-gidx-1]);
				int del = s_row1[lidx] + mis_or_ind;
				int ins = s_row1[lidx-1] + mis_or_ind;

				// Write back to our current sliding window row index, set pointer.
				int mat_idx = row_idx-gidx + gidx*(tlen+1);
				if (match >= ins && match >= del) {
					s_row2[lidx] = match;
					mat[mat_idx] = MATCH;
					if (tidx == blockIdx.x - 1)
						col[row_idx] = match;
				}
				else if (ins >= match && ins >= del) {
					s_row2[lidx] = ins;
					mat[mat_idx] = INS;
					if (tidx == blockIdx.x - 1)
						col[row_idx] = ins;
				}
				else {
					s_row2[lidx] = del;
					mat[mat_idx] = DEL;
					if (tidx == blockIdx.x - 1)
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

  // Save last column that the kernel computes
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
