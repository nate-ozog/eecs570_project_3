#ifndef NEEDLETAIL_GENERAL_HPP
#define NEEDLETAIL_GENERAL_HPP

// Includes.
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

// Definitions.
#define SIM_MAT_PATH   "similarity.txt"
#define GAP_SCORE      -1
#define PTR_BITS       2
#define PTRS_PER_ELT   4
#define PTR_MATCH      1
#define PTR_DEL        2
#define PTR_INS        3
#define PTR_ERR        -1

// Similarity matrix constant GPU memory.
__constant__ signed char c_s[16];

// GPU base-pair to value.
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

// GPU get similarity matrix value for A/B.
__device__ signed char cuda_get_sim(char Ai, char Bi) {
  return c_s[cuda_base_to_val(Ai) * 4 + cuda_base_to_val(Bi)];
}

// Converts (i,j) to the transformed output matrix value.
uint32_t ij_to_z(uint32_t i, uint32_t j, uint32_t qlen, uint32_t tlen) {
  uint32_t z = 0;
  uint32_t jpi = j + i;
  if (jpi <= qlen)
    z = jpi * (jpi + 1) / 2 + i;
  else if (jpi <= tlen)
    z = qlen * (qlen + 1) / 2 + (qlen + 1) * (jpi - qlen) + i;
  else
    z = (tlen + 1) * (qlen + 1)
      - (tlen + qlen + 1 - jpi) * (tlen + qlen + 2 - jpi) / 2
      + i - (jpi - tlen);
  // printf("%d\n", z);
  return z;
}

// Performs ceil(dividend / divisor) using integer division.
uint32_t divide_then_round_up(uint32_t dividend, uint32_t divisor) {
  return (dividend - 1) / divisor + 1;
}

// Gets pointer value from 2-bit encoded, compressed, matrix.
int get_ptr_val(uint8_t * ptr_mat, uint32_t i, uint32_t j, uint32_t qlen, uint32_t tlen) {
  if( i < 0 || i > qlen || j < 0 || j > tlen)
    return PTR_ERR;
  uint32_t z_idx = ij_to_z(i, j, qlen, tlen);
  uint32_t byte_idx = divide_then_round_up(z_idx, PTRS_PER_ELT) - 1;
  uint8_t byte_val = ptr_mat[byte_idx];
  uint8_t shift = PTR_BITS * (z_idx % PTRS_PER_ELT);
  uint8_t mask = 0x03;
  return (byte_val & (mask << shift)) >> shift;
}

// Perform the needletail backtrack algorithm on 2bit encoded, compressed, matrix.
std::pair<char *, char *> nt_backtrack(
  uint8_t * mat,
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  // If we have to flip our query and reference at the end.
  bool flipped = false;
  if (qlen > tlen) {
    flipped = true;
    char * t_temp = t;
    t = q;
    q = t_temp;
    uint32_t tlen_temp = tlen;
    tlen = qlen;
    qlen = tlen_temp;
  }
  // Now run backtrack algorithm with the assumption that qlen <= tlen.
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
    switch(get_ptr_val(mat, i, j, qlen, tlen)) {
      case PTR_MATCH:
        q_algn = q[i - 1] + q_algn;
        t_algn = t[j - 1] + t_algn;
        --i;
        --j;
      break;
      case PTR_INS:
        q_algn = q[i - 1] + q_algn;
        t_algn = '-' + t_algn;
        --i;
      break;
      case PTR_DEL:
        q_algn = '-' + q_algn;
        t_algn = t[j - 1] + t_algn;
        --j;
      break;
      default:
        std::cout << "ERROR, unexpected backtrack-pointer value "
           << get_ptr_val(mat, i, j, qlen, tlen) << " at ("
          << i << "," << j << ") with tlen = " << tlen
          << " and qlen = " << qlen << std::endl;
        exit(-1);
      break;
    }
  }
  // Copy target alignment to c-string.
  char * t_algn_c_str = new char [t_algn.length() + 1];
  std::strcpy(t_algn_c_str, t_algn.c_str());
  // Copy query alignment to c-string.
  char * q_algn_c_str = new char [q_algn.length() + 1];
  std::strcpy(q_algn_c_str, q_algn.c_str());
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