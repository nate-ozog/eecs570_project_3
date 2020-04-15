#ifndef NEEDLETAIL_KERNELS_CUH
#define NEEDLETAIL_KERNELS_CUH

#include "needletail_general.hpp"
#include "testbatch.hpp"
#include "cuda_error_check.cuh"


__global__ void needletail_kernel (
char * t,
char * q,
uint32_t tlen,
uint32_t qlen,
int * col,
uint32_t max_strides,
signed char mis_or_ind,
uint8_t * mat
) {
	// Shorthand thread index.
	uint32_t tidx = threadIdx.x;

	// Set up shared memory row pointers.
	extern __shared__ int smem[];
	int * s_row0 = smem;
	int * s_row1 = s_row0 + (BLOCK_SIZE+1)*max_strides;
	int * s_row2 = s_row1 + (BLOCK_SIZE+1)*max_strides;
	int * s_temp = NULL;

	for (uint32_t col_offset = 1; col_offset < qlen+1; col_offset+=max_strides*BLOCK_SIZE) {

		// Initialize Shared Memory (top two rows of sheared matrix section).
		if (tidx == 0) {
			s_row0[0] = (col_offset-1) * mis_or_ind;// on diag or from previous kernel
			s_row1[0] = col[col_offset];		 	// passed from previous kernel
			s_row1[1] = col_offset * mis_or_ind; 	// always on diag
		}
		__syncthreads();

		// Loop through every row that has useful work.
		for (uint32_t row_idx = col_offset+1; row_idx < qlen+tlen+1; row_idx++) {

			// Calculate row offset, due to diagonalization past row tlen.
			uint32_t row_offset = row_idx <= tlen ? 0 : row_idx - (tlen+1);

			// Calculate last column computed by kernel (local and global).
			uint32_t last_gcol = min(qlen, col_offset+max_strides*BLOCK_SIZE-1);
			uint32_t last_lcol = min(max_strides*BLOCK_SIZE-1, last_gcol-col_offset);

			// Get output matrix offset for current row.
			uint32_t mat_idx;
			if (row_idx <= qlen)
				mat_idx = (row_idx * (row_idx + 1)) / 2 + col_offset;
			else if (row_idx <= tlen)
				mat_idx = (qlen * (qlen + 1)) / 2 + 
					(qlen + 1) * (row_idx - qlen) + col_offset;
			else
				mat_idx = (tlen + 1) * (qlen + 1) - 
					((tlen + qlen + 1 - row_idx) * 
					(tlen + qlen + 2 - row_idx)) / 2 +
					col_offset - row_offset - 1;

			// Set Shared Memory values for leftmost column and diagonal
			if (tidx == 0) {
				s_row2[0] = col[row_idx];
			if (row_idx <= last_gcol)
				s_row2[row_idx-col_offset+1] = row_idx * mis_or_ind;
			}

			// Stride across columns.
			for (uint32_t col_idx = 0; col_idx <= last_lcol; col_idx += BLOCK_SIZE) {

				// Calculate global and local indices for current thread.
				uint32_t gidx = col_offset + col_idx + tidx;
				uint32_t lidx = col_idx + tidx;

				// Do the NW cell calculation.
				if (gidx > row_offset && gidx <= last_gcol && gidx < row_idx) {
					int match = s_row0[lidx] + cuda_nw_get_sim(q[gidx-1], t[row_idx-gidx-1]);
					int del = s_row1[lidx+1] + mis_or_ind;
					int ins = s_row1[lidx] + mis_or_ind;

					// Write back to our current sliding window row index, set pointer.
					if (match >= ins && match >= del) {
						s_row2[lidx+1] = match;
						mat[mat_idx + col_idx + tidx] = MATCH;
						if (lidx == last_lcol)
							col[row_idx] = match;
					}
					else if (ins >= match && ins >= del) {
						s_row2[lidx+1] = ins;
						mat[mat_idx + col_idx + tidx] = INS;
						if (lidx == last_lcol)
							col[row_idx] = ins;
					}
					else {
						s_row2[lidx+1] = del;
						mat[mat_idx + col_idx + tidx] = DEL;
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
}


#endif
