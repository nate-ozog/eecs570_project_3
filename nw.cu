#include "nw_general.h"
#include "xs.cuh"
#include "testbatch.hpp"
#include <pthread.h>

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

// Arguments for worker threads.
struct t_args {
  TestBatch * test_batch;
  Test_t test;
  bool running;
};

// Worker thread function.
void * worker(void * arg) {
  struct t_args * args = (struct t_args *) arg;
  auto start = std::chrono::high_resolution_clock::now();
  uint8_t * nw_ptr_mat = xs_man(args->test.s1, args->test.s2, args->test.s1_len,args->test.s2_len, GAP_SCORE);
  std::pair<char *, char *> algn = nw_ptr_backtrack(nw_ptr_mat, args->test.s1, args->test.s2, args->test.s1_len, args->test.s2_len);
  auto finish = std::chrono::high_resolution_clock::now();
  auto test_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  cuda_error_check(cudaFreeHost(nw_ptr_mat));
  // TODO: CAPTURE OUTPUT SCORE OVER ............................ HERE v
  args->test_batch->log_result(args->test.id, algn.first, algn.second, 0, test_runtime.count());
  delete [] algn.first;
  delete [] algn.second;
  args->running = false;
  return NULL;
}

// Driver.
int main() {
  // Setup system.
  signed char * sim_mat = init_similarity_matrix();
  TestBatch test_batch("batch.txt");
  Test_t test;
  test_batch.set_matrix(sim_mat);
  test_batch.set_gapscore(GAP_SCORE);
  delete [] sim_mat;
  auto start = std::chrono::high_resolution_clock::now();

  // Create N threads.
	int NUM_THREADS = 128;
	pthread_t t[NUM_THREADS];
  struct t_args t_args[NUM_THREADS];
  // Initialize the running flag to null
  for (uint32_t i = 0; i < NUM_THREADS; ++i)
    t_args[i].running = false;

  // Outer loop that runs until all tests are done.
  while (test_batch.next_test(test)) {
    bool test_used = false;
    // Spin wait until we can use this test.
    while (!test_used) {
      // Try to find a home for our test.
      for (uint32_t i = 0; i < NUM_THREADS; ++i) {
        if (!t_args[i].running) {
          t_args[i].test_batch = &test_batch;
          t_args[i].test = test;
          t_args[i].running = true;
          pthread_create(&t[i], NULL, worker, &t_args[i]);
          pthread_detach(t[i]);
          test_used = true;
          break;
        }
      }
    }
  }

  // Make sure all threads are done.
  bool done = false;
  while (!done) {
    done = true;
    for (uint32_t i = 0; i < NUM_THREADS; ++i)
      if (t_args[i].running)
        done = false;
  }

  // Write results and terminate.
  auto finish = std::chrono::high_resolution_clock::now();
  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}