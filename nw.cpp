#include "nw_general.h"
#include "testbatch.hpp"

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
      int match = mat[(tlen+1) * (i-1) + (j-1)] + nw_get_sim(s, q[i-1], t[j-1]);
      int del = mat[(tlen+1) * (i-1) + j] + mis_or_ind;
      int ins = mat[(tlen+1) * i + (j-1)] + mis_or_ind;
      int cell = match > del ? match : del;
      cell = cell > ins ? cell : ins;
      mat[(tlen+1) * i + j] = cell;
    }
  }

  // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
  // for (int i = 0; i <= qlen; ++i) {
  //   for (int j = 0; j <= tlen; ++j)
  //     std::cout << std::setfill(' ') << std::setw(5)
  //       << mat[(tlen+1) * i + j] << " ";
  //   std::cout << std::endl;
  // }

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
    std::pair<char *, char *> algn = nw_backtrack(nw_score_mat, sim_mat, test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE);
    finish = std::chrono::high_resolution_clock::now();
    test_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    total_runtime += test_runtime;
    delete [] nw_score_mat;
    // TODO: CAPTURE OUTPUT SCORE OVER ............... HERE v
    test_batch.log_result(test.id, algn.first, algn.second, 0, test_runtime.count());
    delete [] algn.first;
    delete [] algn.second;
  }

  // Write results and terminate.
  delete [] sim_mat;
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("CPU_results.txt");
  return 0;
}