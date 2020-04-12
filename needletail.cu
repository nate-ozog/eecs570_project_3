#include "nw_general.h"
#include "needletail_threads.cuh"
#include "testbatch.hpp"
#include <pthread.h>

TestBatch test_batch("batch.txt");
volatile uint32_t oustanding_batches = 0;

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
  // Setup system.
  signed char * sim_mat = init_similarity_matrix();
  test_batch.set_matrix(sim_mat);
  test_batch.set_gapscore(GAP_SCORE);
  delete [] sim_mat;

  auto start = std::chrono::high_resolution_clock::now();
  Test_t new_test;
  Test_t ** tests = new Test_t * [STREAM_BATCH_SIZE];
  uint32_t batch_cnt = 0;
  while (test_batch.next_test(new_test)) {
    tests[batch_cnt] = new Test_t;
    *tests[batch_cnt] = new_test;
    ++batch_cnt;
    if (batch_cnt == STREAM_BATCH_SIZE) {
      while (oustanding_batches >= MAX_OUSTANDING_BATCHES);
      needletail_stream_batch(tests);
      batch_cnt = 0;
    }
  }
  while (oustanding_batches);
  auto finish = std::chrono::high_resolution_clock::now();

  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}
