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
#include <mutex>
#include "cuda.h"
#include "cuda_runtime.h"
#include "testbatch.hpp"

#define SIM_MAT_PATH      "similarity.txt"
#define GAP_SCORE         -1
#define MATCH             1
#define DEL               2
#define INS               3
#define NUM_THREADS       16
#define STREAM_BATCH_SIZE 32
#define BLOCK_SIZE        1024

#define DEVICE_POOL_ALLOC_BYTES 2000000000  // Number of bytes per cudaMalloc pool allocation
#define DEVICE_POOL_ALLOC_COUNT 1           // Number of cudaMalloc pool allocations to perform
#define DEVICE_POOL_ALIGN_POW   9           // 2^9 = 512 -> Align pool mallocs to 512B boundaries
#define   HOST_POOL_ALLOC_BYTES 1000000000  // Number of bytes per cudaHostAlloc pool allocation
#define   HOST_POOL_ALLOC_COUNT 1           // Number of cudaHostAlloc pool allocations to perform
#define   HOST_POOL_ALIGN_POW   9           // 2^9 = 512 -> Align pool mallocs to 512B boundaries

__constant__ signed char c_s[16];
extern TestBatch test_batch;

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

// Performs ceil(dividend / divisor) using integer division.
uint32_t divide_then_round_up(uint32_t dividend, uint32_t divisor) {
  return (dividend - 1) / divisor + 1;
}

// Converts (i,j) to "z" compressed format index.
uint32_t ij_to_z(uint32_t i, uint32_t j, uint32_t tlen, uint32_t qlen) {
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

// Pointer backtracking for compressed matrix.
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
    switch(mat[i * (tlen + 1) + j]) {
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
        std::cout << mat[i * (tlen + 1) + j] << std::endl;
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