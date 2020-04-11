#include "nw_general.h"
#include "xs.cuh"
#include "testbatch.hpp"
#include <pthread.h>

TestBatch test_batch("batch.txt");
pthread_barrier_t barrier;

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
  Test_t test;
  std::chrono::high_resolution_clock::time_point start, end;
  while ( test_batch.next_test( test ) ) {
    start = std::chrono::high_resolution_clock::now();
    bool swap_t_q = test.s2_len > test.s1_len;
    if (swap_t_q) {
      std::swap(test.s1, test.s2);
      std::swap(test.s1_len, test.s2_len);
    }
    std::pair<uint8_t *, int> nw_res
      = xs_man(test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE);
    std::pair<char *, char *> algn
      = nw_ptr_backtrack(nw_res.first, swap_t_q, test.s1, test.s2, test.s1_len, test.s2_len);
    cudaFreeHost(nw_res.first);
    end = std::chrono::high_resolution_clock::now();
    test_batch.log_result( test.id, algn.first, algn.second, nw_res.second,
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    delete [] algn.first;
    delete [] algn.second;
  }
  return NULL;
}

// Driver.
int main() {
  // Setup system.
  signed char * sim_mat = init_similarity_matrix();
  test_batch.set_matrix(sim_mat);
  test_batch.set_gapscore(GAP_SCORE);
  delete [] sim_mat;

  // Create N threads.
  int NUM_THREADS = 128;
  pthread_t t[NUM_THREADS];

  pthread_barrier_init( &barrier, NULL, NUM_THREADS );

  auto start = std::chrono::high_resolution_clock::now();
  // Launch the worker threads
  for ( int i = 0; i < NUM_THREADS; i++ )
    pthread_create( &t[i], NULL, worker, NULL );
  // Join the worker threads
  for ( int i = 0; i < NUM_THREADS; i++ )
    pthread_join( t[i], NULL );
  auto finish = std::chrono::high_resolution_clock::now();

  // Write results and terminate.
  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}
