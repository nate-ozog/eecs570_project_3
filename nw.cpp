#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include "testbatch.hpp"

#define SIM_MAT_PATH "similarity.txt"
#define GAP_SCORE    -1

// CPU base-pair to value.
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

// CPU get similarity matrix value for A/B.
signed char get_sim(signed char * s, char Ai, char Bi) {
  return s[base_to_val(Ai) * 4 + base_to_val(Bi)];
}

// CPU backtrack.
std::pair<char *, char *> nw_backtrack (
  int * mat,
  signed char * s,
  const char * t,
  const char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && mat[(tlen+1) * i + j] == mat[(tlen+1) * (i-1) + (j-1)] + get_sim(s, q[i-1], t[j-1])) {
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
  // Copy target alignment to c-string.
  char * t_algn_c_str = new char [t_algn.length() + 1];
  std::strcpy (t_algn_c_str, t_algn.c_str());
  // Copy query alignment to c-string.
  char * q_algn_c_str = new char [q_algn.length() + 1];
  std::strcpy (q_algn_c_str, q_algn.c_str());
  // Put alignment results in pair.
  std::pair<char *, char *> algn (t_algn_c_str, q_algn_c_str);
  return algn;
}

// CPU elementary scoring algorithm.
int * nw_scoring (
  signed char * s,
  const char * t,
  const char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  int * mat = new int [(qlen + 1) * (tlen + 1)];
  mat[0] = 0;
  for (uint32_t i = 1; i <= qlen; ++i)
    mat[(tlen + 1) * i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= tlen; ++i)
    mat[i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= qlen; ++i) {
    for (uint32_t j = 1; j <= tlen; ++j) {
      int match = mat[(tlen+1) * (i-1) + (j-1)] + get_sim(s, q[i-1], t[j-1]);
      int del = mat[(tlen+1) * (i-1) + j] + mis_or_ind;
      int ins = mat[(tlen+1) * i + (j-1)] + mis_or_ind;
      int cell = match > del ? match : del;
      cell = cell > ins ? cell : ins;
      mat[(tlen+1) * i + j] = cell;
    }
  }
  return mat;
}

// Reads in the similarity matrix file into memory.
signed char * init_similarity_matrix() {
  std::string input_line;
  std::string sim_file = SIM_MAT_PATH;
  std::ifstream sim_file_stream(sim_file);
  signed char * s = new signed char[16];
  unsigned char sim_cnt = 0;
  while (std::getline(sim_file_stream, input_line)) {
    s[sim_cnt] = std::stoi(input_line);
    ++sim_cnt;
  }
  return s;
}

// Driver.
int main() {
  // Prepare our time recording.
  auto start = std::chrono::high_resolution_clock::now();
  auto finish = std::chrono::high_resolution_clock::now();
  auto test_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

  // Setup system.
  signed char * sim_mat = init_similarity_matrix();
  TestBatch test_batch("batch.txt");
  Test_t test;
  test_batch.set_matrix(sim_mat);
  test_batch.set_gapscore(GAP_SCORE);

  // Run through each test in input batch.
  while (test_batch.next_test(test)) {
    start = std::chrono::high_resolution_clock::now();
    int * nw_score_mat = nw_scoring(sim_mat, test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE);
    int optimal_score = nw_score_mat[(test.s1_len + 1) * (test.s2_len + 1) - 1];
    std::pair<char *, char *> algn = nw_backtrack(nw_score_mat, sim_mat, test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE);
    finish = std::chrono::high_resolution_clock::now();
    test_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    total_runtime += test_runtime;
    delete [] nw_score_mat;
    test_batch.log_result(test.id, algn.first, algn.second, optimal_score, test_runtime.count());
    delete [] algn.first;
    delete [] algn.second;
  }

  // Write results and terminate.
  delete [] sim_mat;
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("CPU_results.txt");
  return 0;
}