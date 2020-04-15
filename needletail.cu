#include "needletail_general.hpp"
#include "needletail_threads.cuh"
#include "testbatch.hpp"
#include "pools.hpp"
#include <pthread.h>

TestBatch test_batch  ("batch.txt");
PoolMan   device_pool (DEVICE_POOL_ALIGN_POW);
PoolMan   host_pool   (HOST_POOL_ALIGN_POW);

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
  bool swap_t_q;
  std::tuple<char *, char *, int> results;
  std::chrono::high_resolution_clock::time_point start, end;

  while ( test_batch.next_test( test ) ) {
    swap_t_q = test.s2_len > test.s1_len;
    if ( swap_t_q ) {
      std::swap(test.s1,     test.s2);
      std::swap(test.s1_len, test.s2_len);
    }

    start = std::chrono::high_resolution_clock::now();
    results = needletail_stream_single( test.s1, test.s2, test.s1_len, test.s2_len, GAP_SCORE, swap_t_q );
    end = std::chrono::high_resolution_clock::now();

    test_batch.log_result( test.id, std::get<0>(results), std::get<1>(results), std::get<2>(results),
                           std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() );
    delete [] std::get<0>(results);
    delete [] std::get<1>(results);
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

  pthread_t t[NUM_THREADS];
  void *device_pool_ptrs[DEVICE_POOL_ALLOC_COUNT] = {0};
  void   *host_pool_ptrs[  HOST_POOL_ALLOC_COUNT] = {0};

  // Allocate the memory pools
  for ( int i = 0; i < DEVICE_POOL_ALLOC_COUNT; i++ ) {
    cuda_error_check( cudaMalloc( &device_pool_ptrs[i], DEVICE_POOL_ALLOC_BYTES ) );
    device_pool.add_pool( device_pool_ptrs[i], DEVICE_POOL_ALLOC_BYTES );
  }
  for ( int i = 0; i < HOST_POOL_ALLOC_COUNT; i++ ) {
    cuda_error_check( cudaHostAlloc( &host_pool_ptrs[i], HOST_POOL_ALLOC_BYTES, cudaHostAllocDefault ) );
    host_pool.add_pool( host_pool_ptrs[i], HOST_POOL_ALLOC_BYTES );
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Launch the worker threads
  for ( uint32_t i = 0; i < NUM_THREADS; i++)
    pthread_create( &t[i], NULL, worker, NULL );

  // Join the worker threads
  for ( uint32_t i = 0; i < NUM_THREADS; i++ )
    pthread_join( t[i], NULL );

  auto finish = std::chrono::high_resolution_clock::now();

  // Free the memory pools
  for ( int i = 0; i < DEVICE_POOL_ALLOC_COUNT; i++ ) {
    cuda_error_check( cudaFree( device_pool_ptrs[i] ) );
  }
  for ( int i = 0; i < HOST_POOL_ALLOC_COUNT; i++ ) {
    cuda_error_check( cudaFreeHost( host_pool_ptrs[i] ) );
  }

  // Print peak pool usages
  std::cout << "Device pool peak B: " << device_pool.get_peak_bytes()  << "\n";
  std::cout << "            size B: " << device_pool.get_total_bytes() << "\n";
  std::cout << "  Host pool peak B: " <<   host_pool.get_peak_bytes()  << "\n";
  std::cout << "            size B: " <<   host_pool.get_total_bytes() << std::endl;

  // Write results and terminate.
  auto total_runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  test_batch.set_time(total_runtime.count());
  test_batch.save_results("GPU_results.txt");
  return 0;
}