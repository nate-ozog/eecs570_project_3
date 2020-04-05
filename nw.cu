#include "nw_general.h"
#include "xs.cuh"
#include "testbatch.hpp"

// Reads in the similarity matrix file into GPU constant memory.
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
  cudaMemcpyToSymbol(c_s, s, 16 * sizeof(signed char));
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
  delete [] sim_mat;

  // Run through each test in input batch.
  while (test_batch.next_test(test)) {
    start = std::chrono::high_resolution_clock::now();
    uint8_t * nw_ptr_mat = xs_man(test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE);
    std::pair<char *, char *> algn = nw_ptr_backtrack(nw_ptr_mat, test.s1, test.s2, test.s1_len, test.s2_len);
    finish = std::chrono::high_resolution_clock::now();
    test_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    total_runtime += test_runtime;
    cuda_error_check(cudaFreeHost(nw_ptr_mat));
    // TODO: CAPTURE OUTPUT SCORE OVER ............... HERE v
    test_batch.log_result(test.id, algn.first, algn.second, 0, test_runtime.count());
    delete [] algn.first;
    delete [] algn.second;
  }

  // Write results and terminate.
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}