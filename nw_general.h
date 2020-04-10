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
#include <mutex>
#include "cuda.h"
#include "cuda_runtime.h"

#define SIM_MAT_PATH "similarity.txt"
#define GAP_SCORE    -1
#define MATCH        1
#define DEL          2
#define INS          3

__constant__ signed char c_s[16];

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

// Pointer backtracking for standard 2D matrix.
std::pair<char *, char *> nw_ptr_backtrack (
  uint8_t * mat,
  bool flipped,
  const char * t,
  const char * q,
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
  // Copy target alignment to c-string.
  char * t_algn_c_str = new char [t_algn.length() + 1];
  std::strcpy (t_algn_c_str, t_algn.c_str());
  // Copy query alignment to c-string.
  char * q_algn_c_str = new char [q_algn.length() + 1];
  std::strcpy (q_algn_c_str, q_algn.c_str());
  // If we had to flip query and target.
  if (flipped) {
    char * t_algn_c_str_temp = t_algn_c_str;
    t_algn_c_str = q_algn_c_str;
    q_algn_c_str = t_algn_c_str_temp;
  }
  // Put alignment results in pair.
  std::pair<char *, char *> algn (t_algn_c_str, q_algn_c_str);
  return algn;
}

#endif
