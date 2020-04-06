#include "nw_general.h"
#include "xs.h"

int main() {
  // Input variables.
  std::string input_line;
  uint32_t tlen = 0;
  uint32_t qlen = 0;
  char * t = NULL;
  char * q = NULL;
  signed char * s = NULL;

  // Read in similarity matrix file.
  std::string sim_file = "datasets/similarity.txt";
  std::ifstream sim_file_stream(sim_file);
  s = new signed char[16];
  unsigned char sim_cnt = 0;
  while (std::getline(sim_file_stream, input_line)) {
    s[sim_cnt] = std::stoi(input_line);
    ++sim_cnt;
  }
  // Write similarity matrix to constant CUDA memory.
  cudaMemcpyToSymbol(c_s, s, 16 * sizeof(signed char));

  // Prepare our time recording.
  auto start = std::chrono::high_resolution_clock::now();
  auto finish = std::chrono::high_resolution_clock::now();
  auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

  // Run through test file.
  for (uint32_t i = 0; i < NUM_TEST_FILES; ++i) {
    std::string test_file = "datasets/" + std::to_string(i) + ".txt";
    std::ifstream test_file_stream(test_file);
    uint32_t test_cnt = 0;
    while (std::getline(test_file_stream, input_line)) {
      if (test_cnt == 0) {
        tlen = std::stoll(input_line);
        t = new char [tlen + 1];
      }
      if (test_cnt == 1) {
        qlen = std::stoll(input_line);
        q = new char [qlen + 1];
      }
      if (test_cnt == 2)
        strcpy(t, input_line.c_str());
      if (test_cnt == 3)
        strcpy(q, input_line.c_str());
      ++test_cnt;
    }

    // Run matrix computation and time runtime.
    start = std::chrono::high_resolution_clock::now();
    uint8_t * nw_ptr_mat = xs_man(t, q, tlen, qlen, GAP_SCORE);
    finish = std::chrono::high_resolution_clock::now();
    runtime += std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    // Debug print pointer matrix.
    /* print_ptr_mat(nw_ptr_mat, t, q, tlen, qlen); */

    // Backtrack through pointer matrix.
    nw_ptr_backtrack(nw_ptr_mat, t, q, tlen, qlen);

    // Clean up memory.
	cuda_error_check( cudaFreeHost(nw_ptr_mat) );
    delete [] q;
    delete [] t;
  }

  // Clean up similarity matrix memory.
  delete [] s;
  // Print out runtime and kill program.
  std::cerr << "Para Runtime: " << runtime.count() << " us" << std::endl;
  return 0;
}
