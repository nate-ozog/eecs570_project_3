#include "nw_general.h"
#include "needletail_threads.cuh"
#include "testbatch.hpp"
#include <pthread.h>

TestBatch test_batch("batch.txt");

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

// Worker thread function.
void * worker(void * arg) {
  Test_t tests[STREAM_BATCH_SIZE];
  bool swap_t_q[STREAM_BATCH_SIZE];
  uint32_t batch_cnt = 0;
  // Loop until all tests are done.
  while (test_batch.next_test(tests[batch_cnt])) {
    swap_t_q[batch_cnt] = tests[batch_cnt].s2_len > tests[batch_cnt].s1_len;
    if (swap_t_q[batch_cnt]) {
      std::swap(tests[batch_cnt].s1, tests[batch_cnt].s2);
      std::swap(tests[batch_cnt].s1_len, tests[batch_cnt].s2_len);
    }
    ++batch_cnt;
    // If we have a batch ready to go.
    if (batch_cnt == STREAM_BATCH_SIZE) {
      needletail_stream_batch(batch_cnt, swap_t_q, tests);
      batch_cnt = 0;
    }
  }
  // If we ran out of tests but still have not a full batch ready.
  if (batch_cnt != 0)
    needletail_stream_batch(batch_cnt, swap_t_q, tests);
  return NULL;
}

// Driver.
int main() {
  // Setup system.
  signed char * sim_mat = init_similarity_matrix();
  test_batch.set_matrix(sim_mat);
  test_batch.set_gapscore(GAP_SCORE);
  delete [] sim_mat;

  pthread_t t[NUM_THREADS];
  auto start = std::chrono::high_resolution_clock::now();
  // Launch the worker threads
  for (uint32_t i = 0; i < NUM_THREADS; i++)
    pthread_create(&t[i], NULL, worker, NULL);
  // Join the worker threads
  for (uint32_t i = 0; i < NUM_THREADS; i++)
    pthread_join(t[i], NULL);
  auto finish = std::chrono::high_resolution_clock::now();

  // Write results and terminate.
  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}
