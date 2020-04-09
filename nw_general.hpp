#ifndef NW_GENERAL_H
#define NW_GENERAL_H

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

#define NUM_TEST_FILES 8
#define GAP_SCORE -1

#define MATCH 1
#define DEL 2
#define INS 3

__constant__ signed char c_s[16];

uint32_t divide_then_round_up(uint32_t dividend, uint32_t divisor) {
	return (dividend - 1) / divisor + 1;
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
  std::cout << q_algn << std::endl << std::endl;
}

// Print score matrix as backtracking pointer matrix.
void print_score_as_ptr_mat(
  int * mat,
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
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
			if (i > 1 && j > 1 && mat[(tlen+1) * (i-1) + j-1] ==
				mat[(tlen+1)*(i-2) + (j-2)] + nw_get_sim(s, q[i-2], t[j-2])) {
				std::cout << '\\' << " ";
			}
			else if (i > 1 && mat[(tlen+1) * (i-1) + j-1] ==
					mat[(tlen+1) * (i-2) + j-1] + mis_or_ind) {
				std::cout << '^' << " ";
			}
			else { std::cout << '<' << " "; }
		}
	}
    std::cout << std::endl;
  }
}

// Print backtracking pointer matrix.
void print_ptr_mat(
  uint8_t * mat,
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
			uint8_t mvmt = mat[(i-1)*(tlen+1) + j-1];
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

// Pointer backtracking for standard 2D matrix.
void nw_ptr_backtrack(
  uint8_t * mat,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen
) {
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
	  switch(mat[i*(tlen+1)+j]) {
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
		  default:
			std::cout << "ERROR, unexpected back-pointer value: ";
			std::cout << mat[i*(tlen+1)+j] << std::endl;
			std::cout << "i: " << i << "\tj: " << j << std::endl;
			std::cout << "tlen: " << tlen << "\tqlen: " << qlen << std::endl;
			exit(-1);
			break;
	  }
  }
  std::cout << t_algn << std::endl;
  std::cout << q_algn << std::endl << std::endl;
}

#endif
