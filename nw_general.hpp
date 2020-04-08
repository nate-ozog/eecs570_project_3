#ifndef NW_GENERAL_HPP
#define NW_GENERAL_HPP

#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include "cuda.h"
#include "cuda_runtime.h"

#define NUM_TEST_FILES 6
#define GAP_SCORE -1

// 2-bit encoding for alignment matrix back-pointers
#define PTR_BITS 2
#define PTRS_PER_ELT 16
#define MATCH 1
#define DEL 2
#define INS 3
#define OOB -1

__constant__ signed char c_s[16];

// Example similarity matrix.
//    A  G  C  T
// A  1 -1 -1 -1
// G -1  1 -1 -1
// C -1 -1  1 -1
// T -1 -1 -1  1

// Example DP Matrix
//            T
//        A  G  C  T
//     A  ..........
//  Q  G  ..........
//     C  ..........
//     T  ..........

int get_ptr_val(uint32_t* ptr_mat, int i, int j, int h, int w) {
	// Bounds checking
	if( i < 0 || i >= h || j < 0 || j >= w)
		return OOB;
	// Find the uintN value at correct location in pointer matrix
	uint32_t val = ptr_mat[int(i * ceil(w / float(PTRS_PER_ELT)) + j / PTRS_PER_ELT)];
	// Use mask and shift to extract PTR_BITS-bit pointer from uintN
	uint32_t mask = pow(2, PTR_BITS) - 1;
	int shift = PTR_BITS * (j % PTRS_PER_ELT);
	return (val & (mask << shift)) >> shift;
}

signed char base_to_val(char B) {
  // Assume 'A' unless proven otherwise.
  signed char ret = 0;
  if (B == 'G')
    ret = 1;
  if (B == 'C')
    ret = 2;
  if (B == 'T')
    ret = 3;
  return ret;
}

signed char nw_get_sim(signed char * s, char Ai, char Bi) {
  return s[base_to_val(Ai) * 4 + base_to_val(Bi)];
}

__device__ signed char cuda_base_to_val(char B) {
  // Assume 'A' unless proven otherwise.
  signed char ret = 0;
  if (B == 'G')
    ret = 1;
  if (B == 'C')
    ret = 2;
  if (B == 'T')
    ret = 3;
  return ret;
}

__device__ signed char cuda_nw_get_sim(char Ai, char Bi) {
  return c_s[cuda_base_to_val(Ai) * 4 + cuda_base_to_val(Bi)];
}

// Backtrack for standard 2D matrix.
void nw_backtrack(
  int * mat,
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && mat[(tlen+1) * i + j] == mat[(tlen+1) * (i-1) + (j-1)] + nw_get_sim(s, q[i-1], t[j-1])) {
      q_algn = q[i-1] + q_algn;
      t_algn = t[j-1] + t_algn;
      --i;
      --j;
    }
    else if (i > 0 && mat[(tlen+1) * i + j] == mat[(tlen+1) * (i-1) + j] + mis_or_ind) {
      q_algn = q[i-1] + q_algn;
      t_algn = '-' + t_algn;
      --i;
    }
    else {
      q_algn = '-' + q_algn;
      t_algn = t[j-1] + t_algn;
      --j;
    }
  }
  std::cout << t_algn << std::endl;
  std::cout << q_algn << std::endl;
}

// Pointer backtracking for standard 2D matrix.
void nw_ptr_backtrack(
  uint32_t * mat,
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
	  switch(get_ptr_val(mat, i, j, qlen+1, tlen+1)) {
		  case MATCH:
			  q_algn = q[i-1] + q_algn;
			  t_algn = t[j-1] + t_algn;
			  --i; --j;
			  break;
		  case INS:
			  q_algn = q[i-1] + q_algn;
			  t_algn = '-' + t_algn;
			  --i;
			  break;
		  case DEL:
			  q_algn = '-' + q_algn;
			  t_algn = t[j-1] + t_algn;
			  --j;
			  break;
		  case OOB:
			std::cout << "ERROR, out of bounds!" << std::endl;
			std::cout << "i: " << i << "\tj: " << j << std::endl;
			std::cout << "tlen: " << tlen << "\tqlen: " << qlen << std::endl;
			exit(-1);
			break;
		  default:
			std::cout << "ERROR, unexpected back-pointer value: ";
			std::cout << get_ptr_val(mat, i, j, qlen+1, tlen+1) << std::endl;
			std::cout << "i: " << i << "\tj: " << j << std::endl;
			std::cout << "tlen: " << tlen << "\tqlen: " << qlen << std::endl;
			exit(-1);
			break;
	  }
  }
  std::cout << t_algn << std::endl;
  std::cout << q_algn << std::endl;
}


// Print backtracking pointer matrix
void print_ptr_mat(
  uint32_t * mat,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen
) {
	std::cout << "tlen: " << tlen << std::endl;
	std::cout << "qlen: " << qlen << std::endl;
	std::cout << "template: ";
	for (int i = 0; i < tlen; i++)
		std::cout << t[i];
	std::cout << std::endl;
	std::cout << "query: ";
	for (int i = 0; i < qlen; i++)
		std::cout << q[i];
	std::cout << std::endl;

  for (int i = 0; i <= qlen+1; ++i) {
    for (int j = 0; j <= tlen+1; ++j) {
		if (i == 0) {
		  if (j <= 1)
			  std::cout << "." << " ";
	      else
			  std::cout << t[j-2] << " ";
		}
		else if (j == 0) {
		  if (i == 1)
			  std::cout << "." << " ";
		  else
			  std::cout << q[i-2] << " ";
		}
		else {
			int mvmt = get_ptr_val(mat, i-1, j-1, qlen+1, tlen+1);
			char c;
			switch (mvmt) {
				case INS: c = '^'; break;
				case DEL: c = '<'; break;
				case MATCH: c = '\\'; break;
				default: c = 'X'; break;
			}
		    std::cout << c << " ";
		}
	}
    std::cout << std::endl;
  }
}

#endif
